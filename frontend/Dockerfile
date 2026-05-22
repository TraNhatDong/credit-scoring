FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first (layer caching)
COPY package.json ./

# Install dependencies
RUN npm install

# Copy source
COPY . .

# Build Angular (Angular uses environment.prod.ts automatically in production)
RUN npx ng build --configuration production


# ── Stage 2: Nginx Runtime ──────────────────────────────
FROM nginx:alpine AS runtime

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built Angular dist from builder
COPY --from=builder /app/dist/credit-scoring/browser /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost/health || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
