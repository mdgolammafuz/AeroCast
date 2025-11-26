# Stage 1: Builder
FROM golang:1.21-alpine AS builder
WORKDIR /app

# Copy go.mod first for caching (if you have a go.sum, copy it too)
COPY go.mod .
# COPY go.sum .  <-- If you had one, you'd uncomment this

# Copy main.go BEFORE running tidy, so tidy sees the imports!
COPY main.go .

# Now run tidy to fetch the dependencies found in main.go
RUN go mod tidy

# Build the static binary
RUN CGO_ENABLED=0 GOOS=linux go build -o server main.go

# Stage 2: Runner (Distroless/Scratch for tiny image)
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/server .
EXPOSE 8081
CMD ["./server"]