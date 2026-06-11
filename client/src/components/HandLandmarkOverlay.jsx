import { useEffect, useRef } from "react";

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

export default function HandLandmarkOverlay({ videoRef, landmarks }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;

    const { width, height } = video.getBoundingClientRect();
    if (width === 0 || height === 0) return;

    canvas.width  = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    if (!landmarks || landmarks.length === 0) return;

    for (const hand of landmarks) {
      if (!hand || hand.length < 21) continue;

      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth   = 2;
      ctx.lineCap     = "round";

      for (const [a, b] of HAND_CONNECTIONS) {
        const pa = hand[a];
        const pb = hand[b];
        if (!pa || !pb) continue;
        ctx.beginPath();
        ctx.moveTo((1 - pa.x) * width, pa.y * height);
        ctx.lineTo((1 - pb.x) * width, pb.y * height);
        ctx.stroke();
      }

      for (const pt of hand) {
        if (!pt) continue;
        ctx.beginPath();
        ctx.arc((1 - pt.x) * width, pt.y * height, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#ef4444";
        ctx.fill();
      }
    }
  }, [landmarks, videoRef]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}
