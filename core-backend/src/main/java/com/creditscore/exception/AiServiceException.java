package com.creditscore.exception;

public class AiServiceException extends RuntimeException {
    private final int statusCode;

    public AiServiceException(String message) {
        super(message);
        this.statusCode = 503;
    }

    public AiServiceException(String message, int statusCode) {
        super(message);
        this.statusCode = statusCode;
    }

    public AiServiceException(String message, Throwable cause) {
        super(message, cause);
        this.statusCode = 503;
    }

    public int getStatusCode() {
        return statusCode;
    }
}
