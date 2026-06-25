import { formatMatrix } from '../utils/matrix.js';

export default function MatrixBlock({ title, matrix }) {
  return (
    <div className="matrix-block">
      <div className="matrix-title">{title}</div>
      <pre>{formatMatrix(matrix)}</pre>
    </div>
  );
}
