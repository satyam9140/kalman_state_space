export default function MetricCard({ label, value, help }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {help && <small>{help}</small>}
    </div>
  );
}
