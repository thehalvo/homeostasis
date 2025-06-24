package com.homeostasis.healing.utils

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.homeostasis.healing.services.HealingSuggestion
import com.intellij.openapi.diagnostic.Logger
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * HTTP client for communicating with the Homeostasis healing server
 */
class HomeostasisApiClient {
    
    companion object {
        private val LOG = Logger.getInstance(HomeostasisApiClient::class.java)
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }
    
    private val gson = Gson()
    private lateinit var httpClient: OkHttpClient
    private var baseUrl = ""
    private var apiKey = ""
    
    fun initialize(serverUrl: String, apiKey: String) {
        this.baseUrl = serverUrl.trimEnd('/')
        this.apiKey = apiKey
        
        httpClient = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
        
        LOG.info("HomeostasisApiClient initialized with server: $baseUrl")
    }
    
    fun dispose() {
        if (::httpClient.isInitialized) {
            httpClient.dispatcher.executorService.shutdown()
            httpClient.connectionPool.evictAll()
        }
    }
    
    /**
     * Analyze code and get healing suggestions
     */
    suspend fun analyzeCode(content: String, language: String, filePath: String): List<HealingSuggestion> {
        return try {
            val requestBody = AnalyzeRequest(content, language, filePath)
            val response = makeRequest("/api/analyze", requestBody)
            
            if (response.isSuccessful) {
                val responseBody = response.body?.string() ?: ""
                val type = object : TypeToken<List<HealingSuggestion>>() {}.type
                gson.fromJson(responseBody, type) ?: emptyList()
            } else {
                LOG.warn("Analysis request failed: ${response.code} - ${response.message}")
                emptyList()
            }
        } catch (e: Exception) {
            LOG.error("Error during code analysis", e)
            emptyList()
        }
    }
    
    /**
     * Submit telemetry data to the server
     */
    suspend fun sendTelemetry(event: String, data: Map<String, Any>) {
        try {
            val requestBody = TelemetryRequest(event, data)
            val response = makeRequest("/api/telemetry", requestBody)
            
            if (!response.isSuccessful) {
                LOG.warn("Telemetry request failed: ${response.code} - ${response.message}")
            }
        } catch (e: Exception) {
            LOG.error("Error sending telemetry", e)
        }
    }
    
    /**
     * Get server health status
     */
    suspend fun getHealth(): Boolean {
        return try {
            val request = Request.Builder()
                .url("$baseUrl/api/health")
                .get()
                .build()
            
            val response = httpClient.newCall(request).execute()
            response.use { it.isSuccessful }
        } catch (e: Exception) {
            LOG.error("Error checking server health", e)
            false
        }
    }
    
    private fun makeRequest(endpoint: String, requestBody: Any): Response {
        val json = gson.toJson(requestBody)
        val body = json.toRequestBody(JSON_MEDIA_TYPE)
        
        val requestBuilder = Request.Builder()
            .url("$baseUrl$endpoint")
            .post(body)
            .header("Content-Type", "application/json")
        
        if (apiKey.isNotEmpty()) {
            requestBuilder.header("Authorization", "Bearer $apiKey")
        }
        
        return httpClient.newCall(requestBuilder.build()).execute()
    }
    
    /**
     * Data classes for API requests
     */
    private data class AnalyzeRequest(
        val content: String,
        val language: String,
        val filePath: String
    )
    
    private data class TelemetryRequest(
        val event: String,
        val data: Map<String, Any>
    )
}