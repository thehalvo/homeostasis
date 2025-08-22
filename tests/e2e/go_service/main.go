package main

import (
    "net/http"
    "os"

    "github.com/gin-gonic/gin"
)

func main() {
    router := gin.Default()
    
    // Get port from environment or use default
    port := os.Getenv("PORT")
    if port == "" {
        port = "8002"
    }
    
    // Health check endpoint
    router.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status":  "healthy",
            "service": "go-service",
        })
    })
    
    // Error endpoint for testing
    router.GET("/error", func(c *gin.Context) {
        // This endpoint will be modified to trigger specific errors
        c.JSON(http.StatusInternalServerError, gin.H{
            "error": "Test error",
        })
    })
    
    // Start server
    router.Run(":" + port)
}