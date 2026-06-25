import { useEffect, useMemo, useState } from 'react';
import CanvasPlot from './components/CanvasPlot.jsx';
import StateSpaceVisual from './components/StateSpaceVisual.jsx';
import { useAnimation } from './contexts/AnimationContext.jsx';
import { buildAutoModel, makeConclusion, runKalmanAnalysis } from './utils/kalman.js';
import { csvToSignal, estimateMeasurementNoise, estimateProcessNoiseScale, parseCsvText } from './utils/signal.js';

const DATASETS = [
  { id: 'ecg100', name: 'ECG 100 dataset', file: 'ecg100.csv', description: 'Small ECG sample for a quick experiment.' },
  { id: 'ecg200', name: 'ECG 200 dataset', file: 'ecg200.csv', description: 'Medium ECG sample for stable observation.' },
  { id: 'ecg300', name: 'ECG 300 dataset', file: 'ecg300.csv', description: 'Large ECG sample for detailed analysis.' },
];

const GRAPH_OPTIONS = [
  { id: 'main', label: 'Raw / Kalman' },
  { id: 'forcedCompare', label: 'Forced vs unforced' },
  { id: 'residual', label: 'Residual error'},
  { id: 'state', label: 'Hidden states' },
  { id: 'gain', label: 'Kalman gain' },
];

const COLORS = {
  main: {
    raw: '#2563eb',
    kalman: '#e11d48',
    prediction: '#7c3aed',
  },
  compare: {
    raw: '#0f766e',
    forcedPrediction: '#f97316',
    unforcedPrediction: '#0891b2',
    forcedModel: '#16a34a',
    unforcedModel: '#dc2626',
  },
  input: {
    forced: '#ca8a04',
    unforced: '#db2777',
  },
  residual: '#9333ea',
  states: ['#2563eb', '#ea580c', '#16a34a'],
  gains: ['#7c3aed', '#0891b2', '#f43f5e'],
};

function formatNumber(value, digits = 4) {
  if (!Number.isFinite(Number(value))) return '-';
  const n = Number(value);
  if (Math.abs(n) >= 10000 || (Math.abs(n) < 0.0001 && n !== 0)) return n.toExponential(2);
  return n.toFixed(digits).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
}

function pickColumn(parsed, names, fallback) {
  if (!parsed) return fallback;
  const lower = parsed.headers.map((h) => String(h).toLowerCase().trim());
  for (const name of names) {
    const idx = lower.findIndex((h) => h === String(name).toLowerCase());
    if (idx >= 0) return idx;
  }
  return fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, Number(value) || 0));
}

function diagnoseNoise(result, noiseR) {
  if (!result) return null;
  const variance = result.stats.measurement?.variance ?? 0;
  const residualVariance = result.stats.residual?.variance ?? 0;
  const reduction = result.stats.residualVarianceReduction ?? 0;
  const measurementNoise = noiseR?.value ?? 0;
  const ratio = variance > 1e-12 ? residualVariance / variance : 1;

  if (reduction >= 20 || ratio < 0.8 || measurementNoise > variance * 0.05) {
    return {
      status: 'Noisy experiment',
      level: 'noise',
      detail: 'The signal has measurable random variation. The Kalman output reduces part of that variation, so the experiment behaves like noise removal.',
    };
  }
  return {
    status: 'Noiseless experiment',
    level: 'noiseless',
    detail: 'The selected ECG window is already fairly smooth. The Kalman filter mostly tracks the signal instead of removing heavy noise.',
  };
}

function mseBetween(a = [], b = []) {
  const n = Math.min(a.length, b.length);
  if (!n) return 0;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < n; i += 1) {
    if (Number.isFinite(a[i]) && Number.isFinite(b[i])) {
      sum += (a[i] - b[i]) ** 2;
      count += 1;
    }
  }
  return count ? sum / count : 0;
}

export default function App() {
  const {
    playing,
    pause,
    toggle,
    index,
    stepBackward,
    stepForward,
    speed,
    setSpeed,
    setRange,
  } = useAnimation();
  const [selectedDataset, setSelectedDataset] = useState('ecg100');
  const [parsedMap, setParsedMap] = useState({});
  const [loadError, setLoadError] = useState('');
  const [runError, setRunError] = useState('');
  const [hasRun, setHasRun] = useState(false);
  const [showConclusion, setShowConclusion] = useState(false);

  const [modelType, setModelType] = useState('unforced');
  const [dimension, setDimension] = useState(2);
  const [startStep, setStartStep] = useState(0);
  const [endStep, setEndStep] = useState(999);
  const [sampleEvery, setSampleEvery] = useState(1);
  const [graphs, setGraphs] = useState({
    main: true,
    forcedCompare: true,
    residual: true,
    state: true,
    gain: false,
  });

  useEffect(() => {
    let alive = true;
    async function loadDatasets() {
      setLoadError('');
      try {
        const loaded = await Promise.all(DATASETS.map(async (item) => {
          const res = await fetch(item.file, { cache: 'no-store' });
          if (!res.ok) throw new Error(`${item.name} file not found in public folder.`);
          const text = await res.text();
          return [item.id, parseCsvText(text)];
        }));
        if (alive) setParsedMap(Object.fromEntries(loaded));
      } catch (err) {
        if (alive) setLoadError(err.message || 'Dataset loading failed.');
      }
    }
    loadDatasets();
    return () => { alive = false; };
  }, []);

  const parsed = parsedMap[selectedDataset] || null;
  const meta = DATASETS.find((d) => d.id === selectedDataset);

  const timeColumn = pickColumn(parsed, ['time_sec', 'time', 'timestamp'], parsed?.detectedTimeColumn ?? null);
  const valueColumn = pickColumn(parsed, ['ECG_I', 'X2:ECG(raw)', 'ecg_raw', 'raw'], parsed?.detectedValueColumn ?? 1);
  const referenceColumn = pickColumn(parsed, ['ECG_I_filtered', 'X2:ECG(filt)', 'ecg_filtered', 'filtered'], parsed?.detectedReferenceColumn ?? null);

  const signalBuild = useMemo(() => {
    if (!parsed) return { signal: null, error: '' };
    try {
      return { signal: csvToSignal(parsed, timeColumn, valueColumn, 1, referenceColumn), error: '' };
    } catch (err) {
      return { signal: null, error: err.message || 'Could not read selected dataset.' };
    }
  }, [parsed, timeColumn, valueColumn, referenceColumn]);
  const signal = signalBuild.signal;

  const maxStep = signal ? signal.values.length - 1 : 0;
  const safeStart = clamp(startStep, 0, Math.max(0, maxStep - 1));
  const safeEnd = clamp(endStep, safeStart + 1, maxStep);
  const safeSampleEvery = Math.max(1, Math.min(100, Number(sampleEvery) || 1));

  const workingSignal = useMemo(() => {
    if (!signal) return null;
    const time = [];
    const values = [];
    const reference = signal.reference ? [] : null;
    for (let i = safeStart; i <= safeEnd; i += safeSampleEvery) {
      const v = signal.values[i];
      if (!Number.isFinite(v)) continue;
      time.push(signal.time[i] ?? i);
      values.push(v);
      if (reference) reference.push(Number.isFinite(signal.reference[i]) ? signal.reference[i] : NaN);
    }
    if (values.length < 2) return null;
    const diffs = [];
    for (let i = 1; i < time.length; i += 1) {
      const d = time[i] - time[i - 1];
      if (Number.isFinite(d) && d > 0) diffs.push(d);
    }
    const avgDt = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : signal.derivedDt * safeSampleEvery;
    return { ...signal, time, values, reference, derivedDt: avgDt, validRows: values.length };
  }, [signal, safeStart, safeEnd, safeSampleEvery]);

  const noiseR = useMemo(() => (workingSignal ? estimateMeasurementNoise(workingSignal) : null), [workingSignal]);
  const noiseQ = useMemo(() => (workingSignal ? estimateProcessNoiseScale(workingSignal, noiseR?.value) : null), [workingSignal, noiseR]);
  const model = useMemo(() => buildAutoModel(dimension, workingSignal?.derivedDt || 1), [dimension, workingSignal?.derivedDt]);

  const experimentBuild = useMemo(() => {
    if (!hasRun || !workingSignal) return { selected: null, forced: null, unforced: null, error: '' };
    const common = {
      signal: workingSignal,
      model,
      processNoise: noiseQ?.value ?? 1e-6,
      measurementNoise: noiseR?.value ?? 1e-6,
      initialCovariance: 10,
      forceType: 'auto',
      maxSteps: workingSignal.values.length,
    };
    try {
      const forced = runKalmanAnalysis({ ...common, modelType: 'forced' });
      const unforced = runKalmanAnalysis({ ...common, modelType: 'unforced' });
      return {
        forced,
        unforced,
        selected: modelType === 'forced' ? forced : unforced,
        error: '',
      };
    } catch (err) {
      return { selected: null, forced: null, unforced: null, error: err.message || 'Experiment failed.' };
    }
  }, [hasRun, workingSignal, model, modelType, noiseQ, noiseR]);

  const result = experimentBuild.selected;
  const conclusion = result ? makeConclusion(result, model, modelType) : [];
  const noiseDiagnosis = diagnoseNoise(result, noiseR);
  const currentIndex = result ? Math.min(index, result.time.length - 1) : 0;
  const resultLength = result?.time?.length ?? 0;
  const resultTimelineKey = resultLength ? `${resultLength}:${result.time[0]}:${result.time[resultLength - 1]}` : '';

  useEffect(() => {
    if (!resultLength) return;
    setRange({ start: 0, end: resultLength - 1, autoPlay: true, resetIndex: true });
  }, [resultTimelineKey, resultLength, setRange]);

  const stateSeries = result
    ? model.names.map((name, index) => ({ name, values: result.states.map((row) => row[index]), color: COLORS.states[index % COLORS.states.length] }))
    : [];
  const gainSeries = result
    ? model.names.map((name, index) => ({ name: `Gain for ${name}`, values: result.gains.map((row) => row[index]), color: COLORS.gains[index % COLORS.gains.length] }))
    : [];
  const modeCompare = experimentBuild.forced && experimentBuild.unforced
    ? {
      predictionGap: mseBetween(experimentBuild.forced.predicted, experimentBuild.unforced.predicted),
      modelGap: mseBetween(experimentBuild.forced.deterministic, experimentBuild.unforced.deterministic),
      forcedModelMse: experimentBuild.forced.stats.modelMse,
      unforcedModelMse: experimentBuild.unforced.stats.modelMse,
    }
    : null;

  function runExperiment() {
    if (!workingSignal) {
      setRunError('Choose a valid time-step range. At least 2 ECG points are required.');
      return;
    }
    setHasRun(true);
    setShowConclusion(false);
    setRunError('');
    setTimeout(() => document.getElementById('experiment-output')?.scrollIntoView({ behavior: 'smooth' }), 40);
  }

  function changeDataset(datasetId) {
    setSelectedDataset(datasetId);
    setHasRun(false);
    setShowConclusion(false);
    setRunError('');
    setStartStep(0);
    setEndStep(999);
    setSampleEvery(1);
    pause();
    setRange({ start: 0, end: 0, resetIndex: true });
  }

  return (
    <main>
      <header className="project-header simple-card">
        {/* <span className="eyebrow">Animated Kalman ECG lab</span> */}
        <h1>State-Space Kalman</h1>
        <div className={`header-experiment-status ${noiseDiagnosis?.level === 'noise' ? 'noisy' : 'clean'}`}>
          <span>Kalman experiment</span>
          <strong>{noiseDiagnosis?.status || 'Noise / Noiseless experiment'}</strong>
        </div>
        {/* <p>Run a dataset experiment and watch prediction, correction, residual, gain, and state movement step by step.</p> */}
      </header>

      <section className="layout">
        <aside className="simple-card controls">
          <h2>Perform experiment</h2>

          <div className="field-group first-field">
            <label htmlFor="dataset-select">Dataset</label>
            <select id="dataset-select" value={selectedDataset} onChange={(e) => changeDataset(e.target.value)}>
              {DATASETS.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
            <p className="hint">{meta?.description}</p>
          </div>

          <div className="field-group">
            <label>Model type</label>
            <div className="segmented">
              <button className={modelType === 'unforced' ? 'active' : ''} onClick={() => setModelType('unforced')}>Unforced</button>
              <button className={modelType === 'forced' ? 'active' : ''} onClick={() => setModelType('forced')}>Forced</button>
            </div>
          </div>

          <div className="field-group">
            <label htmlFor="state-size">State-space size</label>
            <select id="state-size" value={dimension} onChange={(e) => setDimension(Number(e.target.value))}>
              <option value={1}>1 state: level</option>
              <option value={2}>2 states: level + slope</option>
              <option value={3}>3 states: level + slope + curvature</option>
            </select>
          </div>

          <div className="field-group time-box">
            <label>Time-step window</label>
            <div className="row-2">
              <div><span>Start step</span><input type="number" value={startStep} min="0" max={maxStep} onChange={(e) => setStartStep(e.target.value)} /></div>
              <div><span>End step</span><input type="number" value={endStep} min="1" max={maxStep} onChange={(e) => setEndStep(e.target.value)} /></div>
            </div>
            <input className="range" type="range" min="0" max={maxStep || 1} value={safeEnd} onChange={(e) => setEndStep(e.target.value)} />
            <div className="row-2">
              <div><span>Plot every Nth point</span><input type="number" value={sampleEvery} min="1" max="100" onChange={(e) => setSampleEvery(e.target.value)} /></div>
              <div><span>Points shown</span><strong>{workingSignal?.values.length.toLocaleString() || '-'}</strong></div>
            </div>
          </div>

          <div className="field-group">
            <label>Graphs</label>
            <div className="graph-checks">
              {GRAPH_OPTIONS.map((g) => (
                <button key={g.id} onClick={() => setGraphs((old) => ({ ...old, [g.id]: !old[g.id] }))} className={graphs[g.id] ? 'graph-chip active' : 'graph-chip'}>
                  {graphs[g.id] ? 'On' : 'Off'} - {g.label}
                  <small>{g.help}</small>
                </button>
              ))}
            </div>
          </div>

          <div className="field-group playback-panel">
            <label>Global graph speed</label>
            <div className="speed-row">
              <button type="button" onClick={stepBackward} disabled={!result}>-1</button>
              <button type="button" onClick={toggle} disabled={!result}>{playing ? 'Pause' : 'Play'}</button>
              <button type="button" onClick={stepForward} disabled={!result}>+1</button>
            </div>
            <input
              className="range"
              type="range"
              min="1"
              max="160"
              value={speed}
              onChange={(e) => setSpeed(e.target.value)}
            />
            <div className="playback-readout">
              <span>{speed} steps/sec</span>
              <strong>{result ? `Step ${currentIndex + 1} of ${result.time.length}` : 'Run experiment first'}</strong>
            </div>
          </div>

          <button className="run-btn" onClick={runExperiment} disabled={!workingSignal || Boolean(loadError)}>Run experiment</button>
          {(loadError || signalBuild.error || experimentBuild.error || runError) && <p className="error-box">{loadError || signalBuild.error || experimentBuild.error || runError}</p>}
        </aside>

        <section className="content" id="experiment-output">
          {result && (
            <>
              <section className="plots">
                {graphs.main && (
                  <CanvasPlot
                    title={`${modelType === 'forced' ? 'Forced' : 'Unforced'} Kalman output`}
                    xLabel="Time step"
                    yLabel="ECG amplitude"
                    x={result.time}
                    revealIndex={currentIndex}
                    series={[
                      { name: 'Raw ECG', values: result.measurements, color: COLORS.main.raw },
                      { name: `${modelType === 'forced' ? 'Forced' : 'Unforced'} Kalman ECG`, values: result.filtered, color: COLORS.main.kalman },
                      { name: 'One-step prediction', values: result.predicted, color: COLORS.main.prediction },
                      // ...(result.reference ? [{ name: 'Reference filtered ECG', values: result.reference, color: COLORS.reference }] : []),
                    ]}
                  />
                )}

                {graphs.forcedCompare && experimentBuild.forced && experimentBuild.unforced && (
                  <>
                    <section className="simple-card compare-metrics">
                      <div>
                        <span>Prediction gap</span>
                        <strong>{formatNumber(modeCompare?.predictionGap)}</strong>
                      </div>
                      <div>
                        <span>Model response gap</span>
                        <strong>{formatNumber(modeCompare?.modelGap)}</strong>
                      </div>
                      <div>
                        <span>Forced model MSE</span>
                        <strong>{formatNumber(modeCompare?.forcedModelMse)}</strong>
                      </div>
                      <div>
                        <span>Unforced model MSE</span>
                        <strong>{formatNumber(modeCompare?.unforcedModelMse)}</strong>
                      </div>
                    </section>
                    <CanvasPlot
                      title="Forced vs unforced state-space response"
                      xLabel="Time step"
                      yLabel="ECG amplitude"
                      x={result.time}
                      revealIndex={currentIndex}
                      series={[
                        { name: 'Raw ECG', values: result.measurements, color: COLORS.compare.raw, width: 1.8 },
                        { name: 'Forced one-step prediction', values: experimentBuild.forced.predicted, color: COLORS.compare.forcedPrediction },
                        { name: 'Unforced one-step prediction', values: experimentBuild.unforced.predicted, color: COLORS.compare.unforcedPrediction },
                        { name: 'Forced model-only output', values: experimentBuild.forced.deterministic, color: COLORS.compare.forcedModel },
                        { name: 'Unforced model-only output', values: experimentBuild.unforced.deterministic, color: COLORS.compare.unforcedModel },
                      ]}
                    />
                    <CanvasPlot
                      title="Forced input u(k)"
                      xLabel="Time step"
                      yLabel="Input amplitude"
                      x={result.time}
                      revealIndex={currentIndex}
                      height={260}
                      series={[
                        { name: 'Forced input u(k)', values: experimentBuild.forced.force, color: COLORS.input.forced },
                        { name: 'Unforced input u(k)=0', values: experimentBuild.unforced.force, color: COLORS.input.unforced },
                      ]}
                    />
                  </>
                )}

                {graphs.residual && (
                  <CanvasPlot
                    title="Residual error"
                    description="Residual = raw ECG - active Kalman ECG."
                    xLabel="Time step"
                    yLabel="Residual amplitude"
                    x={result.time}
                    revealIndex={currentIndex}
                    series={[{ name: 'Residual error', values: result.residuals, color: COLORS.residual }]}
                  />
                )}

                {graphs.state && (
                  <CanvasPlot
                    title="Hidden state values"
                    description="These are the internal state values used by the Kalman filter at each step."
                    xLabel="Time step"
                    yLabel="State value"
                    x={result.time}
                    revealIndex={currentIndex}
                    series={stateSeries}
                  />
                )}

                {graphs.gain && (
                  <CanvasPlot
                    title="Kalman gain"
                    description="Higher gain means the filter trusts the ECG measurement more."
                    xLabel="Time step"
                    yLabel="Gain value"
                    x={result.time}
                    revealIndex={currentIndex}
                    series={gainSeries}
                  />
                )}
              </section>

              <StateSpaceVisual result={result} model={model} modelType={modelType} stepIndex={currentIndex} />

              <section className="simple-card metrics">
                <h2>Result summary</h2>
                <div className="summary-grid">
                  <div><span>Dataset</span><strong>{meta?.name}</strong></div>
                  <div><span>Analyzed points</span><strong>{result.stats.samples.toLocaleString()}</strong></div>
                  <div><span>Window</span><strong>{safeStart} to {safeEnd}</strong></div>
                  <div><span>Fit MSE</span><strong>{formatNumber(result.stats.fitMse)}</strong></div>
                  <div><span>Prediction MSE</span><strong>{formatNumber(result.stats.predictionMse)}</strong></div>
                </div>
                <button className="conclusion-btn" onClick={() => setShowConclusion((v) => !v)}>{showConclusion ? 'Hide conclusion' : 'Show conclusion'}</button>
                {showConclusion && (
                  <div className="conclusion">
                    {conclusion.map((line, idx) => <p key={idx}>{line}</p>)}
                  </div>
                )}
              </section>
            </>
          )}
        </section>
      </section>
    </main>
  );
}
