# ============================================================
# Multi-stage build for Spring Boot 3 + Java 25
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM eclipse-temurin:21-jdk-jammy AS builder

WORKDIR /app/build

# Install build tools (Debian-based, uses glibc — required for Lombok compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends bash curl && rm -rf /var/lib/apt/lists/*

# Clear any stale Maven cache for Lombok — forces a fresh download
RUN rm -rf /root/.m2/repository/org/projectlombok/lombok

# Copy Maven / Gradle build files
COPY pom.xml ./
COPY mvnw ./
COPY .mvn ./ .mvn/
COPY src ./src

# Build the application
RUN chmod +x ./mvnw && ./mvnw package -DskipTests -B

# Extract layers for better cache utilization
RUN mkdir -p /app/build/extracted \
    && java -Djarmode=layertools -jar target/*.jar extract --destination /app/build/extracted


# ── Stage 2: Runtime ──────────────────────────────────────
FROM eclipse-temurin:21-jre-jammy AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy extracted layers
COPY --from=builder /app/build/extracted/dependencies/ ./
COPY --from=builder /app/build/extracted/spring-boot-loader/ ./
COPY --from=builder /app/build/extracted/snapshot-dependencies/ ./
COPY --from=builder /app/build/extracted/application/ ./

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=20s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8080/actuator/health || exit 1

# Launch Spring Boot
ENTRYPOINT ["java", "org.springframework.boot.loader.launch.JarLauncher"]
