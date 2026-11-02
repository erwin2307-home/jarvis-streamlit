import { useJarvisStore } from "../store";

const BARS = 24;

export function Waveform() {
  const level = useJarvisStore((s) => s.audioLevel); // dBFS, e.g. -120..0
  // Map dBFS [-60,0] → [0,1]
  const normalized = Math.max(0, Math.min(1, (level + 60) / 60));

  return (
    <div style={{
      position: "absolute", top: "50%", left: "20%", right: "20%",
      height: 60, transform: "translateY(-50%)",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      pointerEvents: "none",
    }}>
      {Array.from({ length: BARS }).map((_, i) => {
        const phase = Math.sin((Date.now() / 200) + i) * 0.5 + 0.5;
        const h = Math.max(0.05, normalized * (0.3 + 0.7 * phase));
        return (
          <span key={i} style={{
            width: 3, height: `${h * 100}%`,
            background: "#00ddff",
            boxShadow: "0 0 6px #00ddff",
          }} />
        );
      })}
    </div>
  );
}
