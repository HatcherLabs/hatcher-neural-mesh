FROM rust:1.91-bookworm AS builder

WORKDIR /src
COPY . .
RUN cargo build --release --locked -p hatcher-ux

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 mesh \
    && useradd --system --uid 10001 --gid mesh --home-dir /nonexistent --shell /usr/sbin/nologin mesh

COPY --from=builder /src/target/release/hatcher-ux /usr/local/bin/hatcher-ux

USER 10001:10001
ENV HATCHER_MESH_PORT=3030
ENV HATCHER_MESH_BIND_ADDR=0.0.0.0
ENV HATCHER_MESH_ALLOW_NON_LOOPBACK_BIND=true
EXPOSE 3030
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD curl --fail --silent --show-error http://127.0.0.1:3030/health >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/hatcher-ux"]
CMD ["serve"]
