# Building the Homeostasis VSCode Extension

This guide explains how to build, test, and package the Homeostasis Self-Healing VSCode extension.

## Prerequisites

- Node.js 16.x or higher
- npm or yarn
- Visual Studio Code

## Development Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Compile TypeScript:**
   ```bash
   npm run compile
   ```

3. **Watch for changes during development:**
   ```bash
   npm run watch
   ```

## Testing

1. **Run linting:**
   ```bash
   npm run lint
   ```

2. **Run tests:**
   ```bash
   npm test
   ```

3. **Test in VSCode:**
   - Open this folder in VSCode
   - Press `F5` to launch a new Extension Development Host
   - Test the extension features in the new window

## Building for Production

1. **Create production build:**
   ```bash
   npm run package
   ```

2. **Package as VSIX:**
   ```bash
   npm run package-extension
   ```

   This creates a `.vsix` file that can be installed in VSCode.

## Installation

### From Source
1. Build the extension (see above)
2. In VSCode, run: `Extensions: Install from VSIX...`
3. Select the generated `.vsix` file

### From Marketplace
Once published, install from the VSCode marketplace:
1. Open VSCode
2. Go to Extensions view (`Ctrl+Shift+X`)
3. Search for "Homeostasis Self-Healing"
4. Click Install

## Configuration

After installation, configure the extension:

1. Open VSCode Settings (`Ctrl+,`)
2. Search for "homeostasis"
3. Configure:
   - Server URL (default: `http://localhost:8080`)
   - API key (if required)
   - Enabled languages
   - Healing preferences

## Development Commands

- `npm run compile` - Compile TypeScript
- `npm run watch` - Watch and compile on changes
- `npm run lint` - Run ESLint
- `npm run test` - Run tests
- `npm run package` - Build for production
- `npm run package-extension` - Create VSIX package

## Project Structure

```
├── src/
│   ├── extension.ts           # Main extension entry point
│   ├── services/              # Core services
│   │   ├── configurationManager.ts
│   │   ├── healingService.ts
│   │   ├── settingsSyncService.ts
│   │   └── telemetryService.ts
│   └── providers/             # VSCode providers
│       ├── codeLensProvider.ts
│       ├── diagnosticsProvider.ts
│       └── inlineFixProvider.ts
├── package.json               # Extension manifest
├── tsconfig.json             # TypeScript configuration
├── webpack.config.js         # Webpack configuration
└── README.md                 # User documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **Extension not activating:**
   - Check VSCode version compatibility
   - Verify installation was successful
   - Check Developer Console for errors

2. **Server connection issues:**
   - Verify Homeostasis server is running
   - Check server URL in settings
   - Verify API key (if required)

3. **Performance issues:**
   - Adjust healing delay in settings
   - Disable real-time healing temporarily
   - Check enabled languages list

### Debug Mode

To enable debug logging:
1. Set `NODE_ENV=development` in environment
2. Check VSCode Developer Console for detailed logs
3. Use VSCode debugger with Extension Development Host