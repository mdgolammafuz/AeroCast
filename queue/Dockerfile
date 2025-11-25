# Stage 1: Builder
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod .
# Run go mod tidy to fetch dependencies if go.sum is missing
RUN go mod tidy
COPY main.go .
RUN CGO_ENABLED=0 GOOS=linux go build -o server main.go

# Stage 2: Runner (Distroless/Scratch for tiny image)
FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/server .
EXPOSE 8081
CMD ["./server"]