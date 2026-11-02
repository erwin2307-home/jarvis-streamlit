import { useJarvisStore } from "../store";

// Hex colors only — avoids appending hex-alpha suffix to an rgba() value
const STATE_COLOR: Record<string, string> = {
  idle: "#00ddff",
  listening: "#00ddff",
  transcribe: "#00ddff",
  thinking: "#aa66ff",
  speaking: "#ffaa00",
  error: "#ff3a3a",
};

// Opacity for the idle ring is handled via lower boxShadow intensity, not color alpha
const IDLE_OPACITY = 0.4;

export function ReactorCore() {
  const state = useJarvisStore((s) => s.state);
  const color = STATE_COLOR[state] ?? STATE_COLOR.idle;
  const pulse = state === "listening" || state === "thinking";
  const ringOpacity = state === "idle" ? IDLE_OPACITY : 1;

  return (
    <div style={{
      position: "absolute", inset: 0, display: "grid", placeItems: "center",
      pointerEvents: "none",
    }}>
      <div style={{
        width: "50vmin", height: "50vmin", borderRadius: "50%",
        border: `2px solid ${color}`,
        opacity: ringOpacity,
        boxShadow: `0 0 80px ${color}66, inset 0 0 60px ${color}44`,
        animation: pulse ? "pulse 1.6s ease-in-out infinite" : undefined,
      }} />
      <div style={{
        position: "absolute",
        width: "10vmin", height: "10vmin", borderRadius: "50%",
        background: `radial-gradient(circle, #fff 0%, ${color} 60%, transparent 80%)`,
        boxShadow: `0 0 50px ${color}`,
      }} />
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%      { opacity: 0.7; transform: scale(1.05); }
        }
      `}</style>
    </div>
  );
}
