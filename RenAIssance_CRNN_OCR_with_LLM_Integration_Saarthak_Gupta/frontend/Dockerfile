# ============================================
# RenAIssance OCR — Frontend Dockerfile
# Multi-stage: Node 20 build → Nginx Alpine slim serve
# ============================================

# ---- Stage 1: Build ----
FROM node:20-alpine AS builder

# Do NOT set NODE_ENV=production here — vite and other build tools
# live in devDependencies and must be installed for the build to succeed.
# NODE_ENV=production is set only in the runtime stage.

WORKDIR /app

# Copy manifests first — layer cache for npm install
COPY package.json package-lock.json ./

# Install ALL dependencies (including devDeps) needed for the build.
# BuildKit cache mount keeps ~/.npm warm across builds so rebuilds after a
# lockfile edit don't re-download every tarball.
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline --no-audit --no-fund

# Copy source and build
COPY . .

RUN npm run build


# ---- Stage 2: Serve ----
FROM nginx:1.30-alpine-slim AS runtime

ENV NODE_ENV=production

# Install curl for runtime health checks.
RUN apk add --no-cache curl

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx SPA config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy only the built static assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Non-root nginx
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown nginx:nginx /var/run/nginx.pid

USER nginx

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
