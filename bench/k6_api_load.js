import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate } from "k6/metrics";

export let options = {
  stages: [
    { duration: "10s", target: 10 },
    { duration: "20s", target: 50 },
    { duration: "10s", target: 0 },
  ],
};

const latencyMs = new Trend("latency_ms");
const successRate = new Rate("success_rate");

function makeSequence() {
  const seq = [];
  for (let i = 0; i < 5; i++) {
    seq.push([25.0, 5.0, 101000.0]); // [temp, wind, pressure]
  }
  return seq;
}

export default function () {
  const url = __ENV.AEROCAST_URL || "http://localhost:8000/predict";
  const payload = JSON.stringify({ sequence: makeSequence() });
  const params = { headers: { "Content-Type": "application/json" } };

  const res = http.post(url, payload, params);

  const ok = check(res, {
    "status is 200": (r) => r.status === 200,
  });

  successRate.add(ok);
  latencyMs.add(res.timings.duration);

  sleep(0.1);
}
