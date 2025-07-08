# Next.js Integration Guide

This document provides guidance on integrating the Homeostasis self-healing framework with Next.js applications.

## Overview

The Homeostasis Next.js plugin enables automatic error detection and healing for Next.js applications. It provides support for:

- Data fetching errors (`getServerSideProps`, `getStaticProps`, `getStaticPaths`)
- API route issues
- App Router and Pages Router errors
- Server and Client component interaction problems
- Image optimization and configuration
- Middleware errors
- Vercel deployment issues

## Setup

1. Install the Homeostasis package:

```bash
npm install homeostasis-js
# or
yarn add homeostasis-js
```

2. Configure Homeostasis in your Next.js application:

```javascript
// homeostasis.config.js
module.exports = {
  enabled: process.env.NODE_ENV === 'production', // or always true for development
  plugins: ['nextjs'],
  settings: {
    monitoring: {
      errorThreshold: 1, // Minimum errors before healing attempt
      healingRate: 60000, // Milliseconds between healing attempts
    },
    nextjs: {
      dataFetching: true,
      apiRoutes: true,
      appRouter: true,
      pagesRouter: true,
      imageOptimization: true,
      middleware: true,
      deployment: true
    }
  }
};
```

3. Add Homeostasis to your Next.js application:

```javascript
// pages/_app.js or app/layout.js
import { initHomeostasis } from 'homeostasis-js';
import homeostasisConfig from '../homeostasis.config';

// Initialize Homeostasis
initHomeostasis(homeostasisConfig);

// Rest of your component...
```

## Features

### Data Fetching Error Healing

The plugin detects and fixes common data fetching errors including:

- Missing or malformed return values in `getServerSideProps` and `getStaticProps`
- Incorrect `getStaticPaths` configuration
- Misuse of `revalidate` or `notFound` properties
- URL format issues in fetch requests

Example fix for a common `getServerSideProps` error:

```javascript
// Before: Error - not returning an object with props
export async function getServerSideProps() {
  const data = await fetchData();
  return data;
}

// After: Fixed - proper return structure
export async function getServerSideProps() {
  try {
    const data = await fetchData();
    return {
      props: {
        data,
      },
    };
  } catch (error) {
    return {
      notFound: true,
    };
  }
}
```

### API Route Error Healing

Detects and fixes issues in API routes including:

- Missing responses
- Improper HTTP method handling
- CORS configuration
- Request body parsing
- Headers already sent errors

Example fix for API route without response:

```javascript
// Before: Error - no response sent
export default async function handler(req, res) {
  const data = await processRequest(req.body);
  // Missing response
}

// After: Fixed - proper response
export default async function handler(req, res) {
  try {
    const data = await processRequest(req.body);
    return res.status(200).json({ data });
  } catch (error) {
    return res.status(500).json({ error: error.message });
  }
}
```

### App Router Error Healing

Detects and fixes issues with the Next.js App Router including:

- Client components importing Server components
- Using Server hooks in Client components
- Route conflicts
- Metadata configuration issues
- Layout nesting problems
- Loading and error boundary implementation

Example fix for client component importing server component:

```javascript
// Before: Error - Client component importing Server component
'use client'
import ServerComponent from './ServerComponent';

// After: Fixed - Using children pattern
// Parent.js (Server Component)
import ClientComponent from './ClientComponent';

export default function Parent() {
  const serverData = fetchServerData();
  
  return (
    <ClientComponent data={serverData}>
      <p>Server content here</p>
    </ClientComponent>
  );
}

// ClientComponent.js
'use client'
export default function ClientComponent({ data, children }) {
  // Client component can use data and render server children
  return (
    <div>
      <h1>Client component with {data}</h1>
      {children}
    </div>
  );
}
```

### Image Optimization Healing

Detects and fixes issues with Next.js Image component and optimization including:

- Missing or incorrect domain configuration
- Missing width and height properties
- Improper loader configuration
- Invalid placeholder usage

Example fix for image domain configuration:

```javascript
// Before: Error - Missing domain configuration
// next.config.js
module.exports = {
  // No image configuration
}

// After: Fixed - Proper image configuration
// next.config.js
module.exports = {
  images: {
    domains: ['example.com', 'cdn.example.com'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920],
    imageSizes: [16, 32, 48, 64, 96, 128, 256]
  }
}
```

### Middleware Error Healing

Detects and fixes issues with Next.js middleware including:

- Improper middleware configuration
- Missing matcher patterns
- Response handling errors
- Middleware chain issues

Example fix for middleware configuration:

```javascript
// Before: Error - Improper middleware configuration
// middleware.js
export function middleware(request) {
  // Missing return or next()
}

// After: Fixed - Proper middleware with matcher and response
// middleware.js
import { NextResponse } from 'next/server';

export function middleware(request) {
  // Check auth for protected routes
  if (request.nextUrl.pathname.startsWith('/dashboard')) {
    const token = request.cookies.get('token')?.value;
    
    if (!token) {
      return NextResponse.redirect(new URL('/login', request.url));
    }
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: ['/dashboard/:path*', '/api/:path*'],
};
```

## Best Practices

1. **Error Monitoring Setup**: Configure proper error monitoring to capture all Next.js errors.

2. **Test Coverage**: Maintain test coverage to verify healing effectiveness.

3. **Staged Rollout**: Use the `healingRate` configuration to control how aggressively fixes are applied.

4. **Custom Rule Extensions**: Extend the default rule set for your specific application needs.

5. **Environment Configuration**: Consider using different healing settings for development vs. production.

## Troubleshooting

If you encounter issues with the Next.js integration:

1. Check logs for detailed error information
2. Verify your Next.js version is supported (12.x and 13.x)
3. Ensure Homeostasis initialization occurs before any application code
4. Check if error is related to custom configurations or third-party libraries

## Advanced Configuration

For advanced scenarios, you can customize the error detection rules:

```javascript
// homeostasis.config.js
module.exports = {
  // ... other configuration
  customRules: {
    nextjs: [
      {
        id: "custom_nextjs_rule",
        pattern: "(?:your custom error pattern)",
        suggestion: "Fix suggestion for your custom error",
        fix_commands: [
          "Step 1 to fix",
          "Step 2 to fix"
        ]
      }
    ]
  }
};
```

## Version Compatibility

The Next.js plugin supports:

- Next.js 12.x and 13.x
- Both Pages Router and App Router
- Server Components in Next.js 13
- Image Optimization v1 and v2
- Middleware API from Next.js 12.0.0 onwards
- Vercel deployment environments

## Contributing

Contributions to improve the Next.js plugin are welcome! See the [contributing guide](./CONTRIBUTING.md) for more information.