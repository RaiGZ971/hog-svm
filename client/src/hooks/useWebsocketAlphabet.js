import { useRef, useCallback } from "react";

export function useWebSocket(setDetectedGesture) {
  const wsRef = useRef(null);

  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/infer");
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDetectedGesture(data);
    };

    ws.onerror = (err) => {
      console.log("WebSocket error", err);
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
    };
  }, [setDetectedGesture]);

  // ✅ ADD THIS
  const sendFrame = useCallback((base64Frame) => {
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({
        action: "predict",
        frame: base64Frame
      }));
    }
  }, []);

  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  return {
    connectWebSocket,
    disconnectWebSocket,
    sendFrame,   // ✅ important
    wsRef,
  };
}
