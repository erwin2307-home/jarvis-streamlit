import { describe, it, expect, beforeEach } from "vitest";
import { render, fireEvent } from "@testing-library/react";
import { App } from "../App";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;
  sent: string[] = [];
  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }
  send(s: string) { this.sent.push(s); }
  close() { this.readyState = 3; this.onclose?.(); }
  simulateOpen() { this.readyState = 1; this.onopen?.(); }
}

describe("App keyboard push-to-talk", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    (globalThis as any).WebSocket = MockWebSocket;
  });

  it("sends manual_start when the space bar is pressed", () => {
    render(<App />);
    MockWebSocket.instances[0].simulateOpen();

    fireEvent.keyDown(window, { key: " ", code: "Space" });

    expect(MockWebSocket.instances[0].sent).toContain('{"type":"manual_start"}');
  });

  it("does not re-trigger on key-repeat while space is held", () => {
    render(<App />);
    MockWebSocket.instances[0].simulateOpen();

    fireEvent.keyDown(window, { key: " ", code: "Space" });
    fireEvent.keyDown(window, { key: " ", code: "Space", repeat: true });

    expect(MockWebSocket.instances[0].sent.length).toBe(1);
  });
});
