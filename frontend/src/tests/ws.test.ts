import { describe, it, expect, vi, beforeEach } from "vitest";
import { JarvisWs } from "../ws";

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
  simulateMessage(payload: object) { this.onmessage?.({ data: JSON.stringify(payload) }); }
}

describe("JarvisWs", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    (globalThis as any).WebSocket = MockWebSocket;
  });

  it("connects and reports open", async () => {
    const events: any[] = [];
    const ws = new JarvisWs({ url: "ws://test", onEvent: (e) => events.push(e), onConnected: () => events.push("OPEN") });
    ws.connect();
    MockWebSocket.instances[0].simulateOpen();
    expect(events).toContain("OPEN");
  });

  it("dispatches parsed events", () => {
    const events: any[] = [];
    const ws = new JarvisWs({ url: "ws://test", onEvent: (e) => events.push(e) });
    ws.connect();
    MockWebSocket.instances[0].simulateOpen();
    MockWebSocket.instances[0].simulateMessage({ type: "state", value: "listening" });
    expect(events[0]).toEqual({ type: "state", value: "listening" });
  });

  it("sends manual_start", () => {
    const ws = new JarvisWs({ url: "ws://test", onEvent: () => {} });
    ws.connect();
    MockWebSocket.instances[0].simulateOpen();
    ws.send({ type: "manual_start" });
    expect(MockWebSocket.instances[0].sent[0]).toBe('{"type":"manual_start"}');
  });

  it("reconnects after close", async () => {
    vi.useFakeTimers();
    const ws = new JarvisWs({ url: "ws://test", onEvent: () => {}, reconnectMs: 100 });
    ws.connect();
    MockWebSocket.instances[0].simulateOpen();
    MockWebSocket.instances[0].close();
    vi.advanceTimersByTime(150);
    expect(MockWebSocket.instances.length).toBe(2);
    vi.useRealTimers();
  });
});
