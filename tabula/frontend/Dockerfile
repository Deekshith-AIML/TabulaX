# ---- TabulaX frontend image (build with Vite, serve with nginx) ----

# Stage 1: build static assets
FROM node:20-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# VITE_API_URL is read at BUILD time and inlined into the bundle by Vite.
# The browser calls this origin directly, so it must be reachable from the
# user's browser (e.g. http://localhost:8000 for local compose).
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

# Stage 2: serve with nginx
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

# 127.0.0.1 (not "localhost"): busybox wget resolves ::1 first and does not
# fall back to IPv4, while nginx listens on IPv4 only.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null 2>&1 || exit 1
