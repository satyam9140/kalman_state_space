import {
  add,
  identity,
  matrixToVector,
  meanSquaredError,
  multiply,
  safeInverse,
  scalarMultiply,
  subtract,
  symmetrize,
  transpose,
  vectorStats,
  vectorToMatrix,
} from './matrix.js';
import { generateForce, savitzkyGolaySmooth } from './signal.js';

export function buildAutoModel(dimension = 2, dt = 1) {
  const dim = Math.max(1, Math.min(3, Number(dimension) || 2));
  const step = Math.max(1e-12, Number(dt) || 1);

  if (dim === 1) {
    return {
      A: [[1]],
      B: [[step]],
      C: [[1]],
      D: [[0]],
      names: ['level'],
      description: '1-state random-walk model: the state is the ECG level, with optional input changing the level over time.',
    };
  }

  if (dim === 3) {
    return {
      A: [
        [1, step, 0.5 * step * step],
        [0, 1, step],
        [0, 0, 1],
      ],
      B: [[(step ** 3) / 6], [0.5 * step * step], [step]],
      C: [[1, 0, 0]],
      D: [[0]],
      names: ['level', 'velocity/slope', 'acceleration/curvature'],
      description: '3-state constant-acceleration model: level, slope, and curvature are estimated while input behaves like jerk.',
    };
  }

  return {
    A: [
      [1, step],
      [0, 1],
    ],
    B: [[0.5 * step * step], [step]],
    C: [[1, 0]],
    D: [[0]],
    names: ['level', 'velocity/slope'],
    description: '2-state constant-velocity model: level and slope are estimated while input behaves like acceleration.',
  };
}

function isMatrixShape(matrix, rows, cols) {
  return Array.isArray(matrix)
    && matrix.length === rows
    && matrix.every((row) => Array.isArray(row) && row.length === cols && row.every((value) => Number.isFinite(Number(value))));
}

export function validateModel({ A, B, C, D }) {
  if (!Array.isArray(A) || !A.length || A.some((row) => !Array.isArray(row) || row.length !== A.length)) {
    throw new Error('A must be a square n × n matrix.');
  }
  const n = A.length;
  if (!isMatrixShape(A, n, n)) throw new Error('A contains invalid numbers.');
  if (!isMatrixShape(B, n, 1)) throw new Error('B must be n × 1.');
  if (!isMatrixShape(C, 1, n)) throw new Error('C must be 1 × n for single-output signal analysis.');
  if (!isMatrixShape(D, 1, 1)) throw new Error('D must be 1 × 1.');
}

function initialStateFromMeasurement(z0, dimension) {
  const x = Array(dimension).fill(0);
  x[0] = Number.isFinite(z0) ? z0 : 0;
  return vectorToMatrix(x);
}

function scalarOutput(C, x, D, u) {
  return multiply(C, x)[0][0] + D[0][0] * u;
}

function finiteOr(value, fallback = 0) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

function cleanReference(reference, total) {
  if (!reference || !reference.some(Number.isFinite)) return null;
  return reference.slice(0, total).map((value) => (Number.isFinite(value) ? Number(value) : NaN));
}

function buildDriverSeries(signal, total) {
  const measured = signal.values.slice(0, total).map((value, index) => finiteOr(value, index ? signal.values[index - 1] : 0));
  const reference = cleanReference(signal.reference, total);
  const source = reference || savitzkyGolaySmooth(measured, measured.length >= 9 ? 9 : 5);
  const driver = [];

  for (let i = 0; i < total; i += 1) {
    const fallback = measured[i] ?? driver[i - 1] ?? 0;
    driver.push(finiteOr(source[i], fallback));
  }

  return driver;
}

function regularizedLeastSquaresInput(model, state, target, lambda) {
  const naturalState = multiply(model.A, state);
  const naturalOutput = scalarOutput(model.C, naturalState, model.D, 0);
  const outputGain = multiply(model.C, model.B)[0][0] + model.D[0][0];
  const residual = target - naturalOutput;
  const denominator = outputGain * outputGain + lambda;
  const u = denominator > 1e-18 ? (outputGain * residual) / denominator : 0;
  return {
    u: Number.isFinite(u) ? u : 0,
    naturalState,
  };
}

function buildAutoForce(signal, model, total) {
  const driver = buildDriverSeries(signal, total);
  const u = [];
  const n = model.A.length;
  let xModel = initialStateFromMeasurement(driver[0], n);
  const measurementVariance = vectorStats(driver).variance || 1;
  const lambda = Math.max(1e-12, measurementVariance * 1e-8);

  for (let k = 0; k < total; k += 1) {
    const target = driver[k];
    const { u: uk, naturalState } = regularizedLeastSquaresInput(model, xModel, target, lambda);
    u.push(uk);
    xModel = add(naturalState, scalarMultiply(model.B, uk));
  }

  return u;
}

function buildProcessNoiseMatrix(dimension, dt, noiseScale) {
  const q = Math.max(1e-12, Number(noiseScale) || 1e-12);
  const t = Math.max(1e-12, Number(dt) || 1);

  if (dimension === 1) return [[q * t]];

  if (dimension === 2) {
    return [
      [q * (t ** 3) / 3, q * (t ** 2) / 2],
      [q * (t ** 2) / 2, q * t],
    ];
  }

  return [
    [q * (t ** 5) / 20, q * (t ** 4) / 8, q * (t ** 3) / 6],
    [q * (t ** 4) / 8, q * (t ** 3) / 3, q * (t ** 2) / 2],
    [q * (t ** 3) / 6, q * (t ** 2) / 2, q * t],
  ];
}

function sanitizeMeasurements(values) {
  const out = [];
  for (let i = 0; i < values.length; i += 1) {
    const current = Number(values[i]);
    if (Number.isFinite(current)) out.push(current);
    else out.push(out[i - 1] ?? 0);
  }
  return out;
}

function covarianceUpdateJoseph(I, K, C, PPred, R) {
  const KC = multiply(K, C);
  const left = subtract(I, KC);
  const right = transpose(left);
  const KRKT = multiply(multiply(K, R), transpose(K));
  return symmetrize(add(multiply(multiply(left, PPred), right), KRKT));
}

export function runKalmanAnalysis({
  signal,
  model,
  modelType = 'unforced',
  processNoise = 0.0001,
  measurementNoise = 0.08,
  initialCovariance = 10,
  forceType = 'sine',
  forceAmplitude = 1,
  forceFrequency = 0.02,
  maxSteps = 0,
}) {
  validateModel(model);
  const n = model.A.length;
  const total = Math.min(signal.values.length, maxSteps && maxSteps > 0 ? maxSteps : signal.values.length);
  if (total < 2) throw new Error('At least two samples are required for Kalman analysis.');

  const dt = Math.max(1e-12, signal.derivedDt || (signal.time.length > 1 ? signal.time[1] - signal.time[0] : 1));
  const measurements = sanitizeMeasurements(signal.values.slice(0, total));
  const reference = cleanReference(signal.reference, total);
  const time = signal.time.slice(0, total);
  const u = modelType === 'forced'
    ? (forceType === 'auto'
      ? buildAutoForce({ ...signal, values: measurements }, model, total)
      : generateForce(forceType, total, dt, forceAmplitude, forceFrequency))
    : Array(total).fill(0);

  const Q = buildProcessNoiseMatrix(n, dt, processNoise);
  const R = [[Math.max(1e-12, Number(measurementNoise) || 0)]];
  let P = identity(n, Math.max(1e-12, Number(initialCovariance) || 1));
  let xHat = initialStateFromMeasurement(measurements[0], n);
  let xModel = initialStateFromMeasurement(measurements[0], n);
  const I = identity(n);
  const AT = transpose(model.A);
  const CT = transpose(model.C);

  const filtered = [];
  const predicted = [];
  const deterministic = [];
  const residuals = [];
  const innovations = [];
  const states = [];
  const gains = [];

  for (let k = 0; k < total; k += 1) {
    const uk = finiteOr(u[k], 0);
    const Bu = scalarMultiply(model.B, uk);

    const xPred = add(multiply(model.A, xHat), Bu);
    const PPred = symmetrize(add(multiply(multiply(model.A, P), AT), Q));
    const yPred = scalarOutput(model.C, xPred, model.D, uk);
    predicted.push(yPred);

    if (k > 0) xModel = add(multiply(model.A, xModel), Bu);
    deterministic.push(scalarOutput(model.C, xModel, model.D, uk));

    const innovationValue = measurements[k] - yPred;
    const innovation = [[innovationValue]];
    const S = add(multiply(multiply(model.C, PPred), CT), R);
    const K = multiply(multiply(PPred, CT), safeInverse(S));
    xHat = add(xPred, multiply(K, innovation));
    P = covarianceUpdateJoseph(I, K, model.C, PPred, R);

    const yFilt = scalarOutput(model.C, xHat, model.D, uk);
    filtered.push(yFilt);
    residuals.push(measurements[k] - yFilt);
    innovations.push(innovationValue);
    states.push(matrixToVector(xHat));
    gains.push(matrixToVector(K));
  }

  const measurementStats = vectorStats(measurements);
  const residualStats = vectorStats(residuals);
  const fitMse = meanSquaredError(measurements, filtered);
  const modelMse = meanSquaredError(measurements, deterministic);
  const predictionMse = meanSquaredError(measurements, predicted);
  const fitRmse = Math.sqrt(fitMse);
  const predictionRmse = Math.sqrt(predictionMse);
  const fitMae = measurements.reduce((sum, value, index) => sum + Math.abs(value - filtered[index]), 0) / total;
  const normalizedFitError = measurementStats.variance > 1e-12 ? fitMse / measurementStats.variance : fitMse;
  const residualVarianceReduction = measurementStats.variance > 1e-12
    ? 100 * (1 - residualStats.variance / measurementStats.variance)
    : 0;
  const innovationStats = vectorStats(innovations);

  let referenceStats = null;
  if (reference) {
    const rawReferenceMse = meanSquaredError(measurements, reference);
    const filteredReferenceMse = meanSquaredError(filtered, reference);
    const modelReferenceMse = meanSquaredError(deterministic, reference);
    const referenceImprovement = rawReferenceMse > 1e-12
      ? 100 * (1 - filteredReferenceMse / rawReferenceMse)
      : 0;
    referenceStats = {
      stats: vectorStats(reference.filter(Number.isFinite)),
      rawReferenceMse,
      filteredReferenceMse,
      modelReferenceMse,
      referenceImprovement,
    };
  }

  return {
    time,
    measurements,
    reference,
    filtered,
    predicted,
    deterministic,
    residuals,
    states,
    gains,
    force: u,
    finalGain: gains[gains.length - 1] || [],
    finalState: states[states.length - 1] || [],
    stats: {
      samples: total,
      measurement: measurementStats,
      residual: residualStats,
      innovation: innovationStats,
      reference: referenceStats,
      fitMse,
      fitRmse,
      fitMae,
      modelMse,
      predictionMse,
      predictionRmse,
      normalizedFitError,
      residualVarianceReduction,
      noiseReduction: residualVarianceReduction,
    },
    matrices: {
      A: model.A,
      B: model.B,
      C: model.C,
      D: model.D,
      Q,
      R,
    },
  };
}

export function makeConclusion(result, model, modelType) {
  const { stats } = result;
  const ratio = stats.normalizedFitError;
  const lines = [];

  if (ratio < 0.08) {
    lines.push('The Kalman estimate tracks the measured signal closely. This is a usable filtering setup, but it still needs reference-based validation before claiming true denoising.');
  } else if (ratio < 0.3) {
    lines.push('The result is usable but not perfect. The model captures the broad motion, while some signal behavior is still unexplained.');
  } else {
    lines.push('The fit is weak. This is not a small tuning issue; the model structure is probably too simple for the uploaded signal. Increase dimension, use forced mode, or use custom matrices.');
  }

  if (modelType === 'unforced') {
    lines.push('Unforced mode means u(k)=0, so B is not actively used. If the deterministic curve is poor, the signal contains behavior that the free state transition A cannot explain by itself.');
  } else {
    lines.push('Forced mode uses B·u(k), so the model tests whether an external input can explain changes in the signal better than free evolution alone.');
  }

  if (model.A.length === 1) lines.push('The 1D model is easiest to understand but can only track the level. It will struggle on fast slopes, ramps, and curved motion.');
  if (model.A.length === 2) lines.push('The 2D model estimates level and slope, which is usually the best beginner baseline for most single-channel time-series.');
  if (model.A.length >= 3) lines.push('The 3D/custom model can explain curvature, but it can also overreact if tracking sensitivity is too aggressive.');

  if (stats.reference) {
    const improvement = stats.reference.referenceImprovement;
    if (improvement > 5) {
      lines.push(`Against the selected reference column, the Kalman estimate improves MSE by about ${improvement.toFixed(1)}% compared with the raw measured signal.`);
    } else if (improvement >= 0) {
      lines.push(`Against the selected reference column, the improvement is only about ${improvement.toFixed(1)}%. The filter smooths the signal, but it is not a strong denoising proof.`);
    } else {
      lines.push(`Against the selected reference column, the Kalman estimate is worse than the raw signal by about ${Math.abs(improvement).toFixed(1)}%. These settings should not be used as a final denoising model.`);
    }
  }

  if (stats.residualVarianceReduction > 0) {
    lines.push(`Residual variance is lower than raw signal variance by about ${stats.residualVarianceReduction.toFixed(1)}%. This means the estimate is smoother than the measurement; it is not automatically proof that real-world noise was removed.`);
  } else {
    lines.push('Residual variance did not improve. That means the filter settings or model are not giving useful smoothing for this signal.');
  }

  return lines;
}
