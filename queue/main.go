package main

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Simple in-memory queue (buffered channel)
var (
	eventQueue = make(chan string, 100)
	lock       sync.Mutex
)

// Metrics
var (
	queueDepth = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "aerocast_queue_depth",
		Help: "Current number of events in the drift queue",
	})
	eventsPublished = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "aerocast_queue_published_total",
		Help: "Total events published to queue",
	})
	eventsConsumed = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "aerocast_queue_consumed_total",
		Help: "Total events consumed from queue",
	})
)

func init() {
	prometheus.MustRegister(queueDepth)
	prometheus.MustRegister(eventsPublished)
	prometheus.MustRegister(eventsConsumed)
}

// POST /publish
// Payload: {"event": "drift"}
func publishHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", 405)
		return
	}

	var msg map[string]string
	if err := json.NewDecoder(r.Body).Decode(&msg); err != nil {
		http.Error(w, "Bad JSON", 400)
		return
	}

	event := msg["event"]
	if event == "" {
		event = "unknown"
	}

	select {
	case eventQueue <- event:
		eventsPublished.Inc()
		queueDepth.Set(float64(len(eventQueue)))
		w.WriteHeader(http.StatusAccepted)
		w.Write([]byte(`{"status":"queued"}`))
		log.Printf("[queue] Event published: %s", event)
	default:
		http.Error(w, "Queue full", 503)
		log.Println("[queue] Drop: Queue full")
	}
}

// GET /subscribe
// Returns 200 {"event": "..."} if event exists, else 204 No Content
func subscribeHandler(w http.ResponseWriter, r *http.Request) {
	select {
	case evt := <-eventQueue:
		eventsConsumed.Inc()
		queueDepth.Set(float64(len(eventQueue)))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"event": evt})
		log.Printf("[queue] Event consumed: %s", evt)
	default:
		w.WriteHeader(http.StatusNoContent)
	}
}

func main() {
	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/publish", publishHandler)
	http.HandleFunc("/subscribe", subscribeHandler)

	log.Println("Go Event Queue listening on :8081")
	// Update depth metric periodically just in case
	go func() {
		for range time.Tick(5 * time.Second) {
			queueDepth.Set(float64(len(eventQueue)))
		}
	}()

	log.Fatal(http.ListenAndServe(":8081", nil))
}