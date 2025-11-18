import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "10s", target: 10 },
    { duration: "20s", target: 50 },
    { duration: "10s", target: 0 },
  ],
};

const BASE_URL = "http://localhost:8000";

// match the FastAPI /predict schema: { "sequence": [[temp, windspeed, pressure], ...] }
function makeSequence(windowSize = 24) {
  const seq = [];
  for (let i = 0; i < windowSize; i++) {
    // use realistic but constant-ish values; model doesn’t care for load test
    seq.push([25.0, 5.0, 101000.0]);
  }
  return seq;
}

export default function () {
  const payload = JSON.stringify({
    sequence: makeSequence(24),
  });

  const params = {
    headers: {
      "Content-Type": "application/json",
    },
  };

  const res = http.post(`${BASE_URL}/predict`, payload, params);

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  sleep(0.1); // tiny pause so we don’t go completely bonkers
}
