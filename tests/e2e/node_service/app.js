const express = require('express');
const app = express();
const port = process.env.PORT || 8001;

app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', service: 'node-service' });
});

// Error endpoint for testing
app.get('/error', (req, res) => {
    // This endpoint will be modified to trigger specific errors
    res.status(500).json({ error: 'Test error' });
});

// Start server
app.listen(port, () => {
    console.log(`Node.js test service listening on port ${port}`);
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});