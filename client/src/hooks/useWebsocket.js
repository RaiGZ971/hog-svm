// import { useRef, useCallback } from "react";
//
// export function useWebSocket(setDetectedGesture) {
//   const wsRef = useRef(null);
//
//   const connectWebSocket = useCallback(() => {
//     const ws = new WebSocket("ws://localhost:8000/ws");
//     wsRef.current = ws;
//
//     ws.onopen = () => {
//       console.log("WebSocket connected");
//     };
//
//     ws.onmessage = (event) => {
//       setDetectedGesture(event.data);
//     };
//
//     ws.onerror = (err) => {
//       console.log("WebSocket error", err);
//     };
//
//     ws.onclose = () => {
//       console.log("WebSocket closed");
//     };
//   }, [setDetectedGesture]);
//
//   const disconnectWebSocket = useCallback(() => {
//     if (wsRef.current) {
//       wsRef.current.close();
//       wsRef.current = null;
//     }
//   }, []);
//
//   return {
//     connectWebSocket,
//     disconnectWebSocket,
//     wsRef,
//   };
// }
//
import { useRef, useCallback, useEffect } from "react";

const WS_URL = "ws://localhost:8000/ws";
// how long a label must be stable (ms) before showing it in the UI
const DEBOUNCE_MS = 400;

export function useWebSocket(setDetectedGesture) {
  const wsRef      = useRef(null);
  const debounceRef = useRef(null);
  const lastLabel  = useRef(null);

  // clean up on unmount
  useEffect(() => {
    return () => {
      clearTimeout(debounceRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const connectWebSocket = useCallback(() => {
    // don't double-connect
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[WS] connected");
    };

    ws.onmessage = (event) => {
      const label = event.data;

      // debounce: only update UI when the same label
      // has been received consistently for DEBOUNCE_MS
      if (label !== lastLabel.current) {
        lastLabel.current = label;
        clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => {
          setDetectedGesture(label);
        }, DEBOUNCE_MS);
      }
    };

    ws.onerror = (err) => {
      console.warn("[WS] error", err);
    };

    ws.onclose = () => {
      console.log("[WS] closed");
    };
  }, [setDetectedGesture]);

  const disconnectWebSocket = useCallback(() => {
    clearTimeout(debounceRef.current);
    lastLabel.current = null;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  return { connectWebSocket, disconnectWebSocket, wsRef };
}
