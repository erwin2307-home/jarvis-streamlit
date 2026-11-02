import { ReactorCore } from "./ReactorCore";
import { Waveform } from "./Waveform";
import { StatusCorners } from "./StatusCorners";

export function HudShell() {
  return (
    <div style={{
      position: "fixed", inset: 0, background: "radial-gradient(ellipse at center, #001218 0%, #000508 100%)",
      overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage:
          "linear-gradient(rgba(0,200,255,0.06) 1px, transparent 1px), " +
          "linear-gradient(90deg, rgba(0,200,255,0.06) 1px, transparent 1px)",
        backgroundSize: "32px 32px",
      }} />
      <ReactorCore />
      <Waveform />
      <StatusCorners />
    </div>
  );
}
