# Vue.js Integration Guide

This guide explains how to use Homeostasis with Vue.js applications, including Vue 2, Vue 3, Nuxt.js, and related frameworks.

## Overview

The Vue.js plugin for Homeostasis provides comprehensive error detection and automatic healing for:

- **Vue Components**: Component lifecycle, props validation, event handling
- **Composition API**: ref, computed, watch, lifecycle hooks, setup function
- **Vuex State Management**: Store configuration, mutations, actions, getters
- **Vue Router**: Route configuration, navigation guards, dynamic routing
- **Vue 3 Features**: Teleport, Suspense, fragments, script setup
- **Template Syntax**: Directives, bindings, expressions, slots

## Supported Frameworks

- Vue.js 2.x and 3.x
- Nuxt.js (Universal Vue Apps)
- Quasar Framework
- Vue CLI projects
- Vite + Vue
- VuePress (documentation sites)
- Gridsome (static site generator)

## Installation and Setup

### Basic Setup

1. **Install Homeostasis** in your Vue project:
```bash
npm install homeostasis-vue
```

2. **Configure monitoring** in your main application file:
```javascript
// main.js (Vue 3)
import { createApp } from 'vue'
import { createHomeostasis } from 'homeostasis-vue'
import App from './App.vue'

const app = createApp(App)

// Initialize Homeostasis
const homeostasis = createHomeostasis({
  framework: 'vue',
  version: '3.x',
  features: ['composition-api', 'vuex', 'router']
})

app.use(homeostasis)
app.mount('#app')
```

### Nuxt.js Setup

For Nuxt.js applications, add the Homeostasis module:

```javascript
// nuxt.config.js
export default {
  modules: [
    ['homeostasis-vue/nuxt', {
      framework: 'nuxt',
      ssr: true,
      features: ['vuex', 'router', 'composition-api']
    }]
  ]
}
```

## Error Detection Capabilities

### 1. Composition API Errors

**Missing Import Errors**
```javascript
// ❌ Error: ref is not defined
export default {
  setup() {
    const count = ref(0) // Error: ref is not defined
  }
}

// ✅ Auto-fix: Add import
import { ref } from 'vue'

export default {
  setup() {
    const count = ref(0)
    return { count }
  }
}
```

**Setup Function Return Issues**
```javascript
// ❌ Error: setup must return an object
export default {
  setup() {
    const count = ref(0)
    // Missing return statement
  }
}

// ✅ Auto-fix: Add return statement
export default {
  setup() {
    const count = ref(0)
    return { count }
  }
}
```

### 2. Vuex State Management

**Store Configuration Issues**
```javascript
// ❌ Error: store is not defined
export default {
  mounted() {
    console.log(this.$store.state.count) // Error: store not configured
  }
}

// ✅ Auto-fix: Configure Vuex store
import { createStore } from 'vuex'

const store = createStore({
  state: {
    count: 0
  },
  mutations: {
    increment(state) {
      state.count++
    }
  }
})

export default store
```

**Direct State Mutation**
```javascript
// ❌ Error: Do not mutate Vuex store state outside mutation handlers
methods: {
  increment() {
    this.$store.state.count++ // Direct mutation - not allowed
  }
}

// ✅ Auto-fix: Use mutations
methods: {
  increment() {
    this.$store.commit('increment')
  }
}
```

### 3. Vue Router Navigation

**Route Configuration Issues**
```javascript
// ❌ Error: route not found
const routes = [
  { path: '/', component: Home }
  // Missing route for /about
]

// ✅ Auto-fix: Add missing route
const routes = [
  { path: '/', component: Home },
  { path: '/about', component: About }
]
```

**Navigation Guard Errors**
```javascript
// ❌ Error: navigation cancelled
router.beforeEach((to, from, next) => {
  if (to.meta.requiresAuth && !isAuthenticated) {
    // Missing next() call
  }
})

// ✅ Auto-fix: Add proper next() handling
router.beforeEach((to, from, next) => {
  if (to.meta.requiresAuth && !isAuthenticated) {
    next('/login')
  } else {
    next()
  }
})
```

### 4. Template Syntax

**Missing Key in v-for**
```vue
<!-- ❌ Error: missing key in v-for -->
<template>
  <li v-for="item in items">{{ item.name }}</li>
</template>

<!-- ✅ Auto-fix: Add unique key -->
<template>
  <li v-for="item in items" :key="item.id">{{ item.name }}</li>
</template>
```

**Component Registration Issues**
```vue
<!-- ❌ Error: component not registered -->
<template>
  <MyButton @click="handleClick">Click me</MyButton>
</template>

<!-- ✅ Auto-fix: Register component -->
<script>
import MyButton from './components/MyButton.vue'

export default {
  components: {
    MyButton
  },
  methods: {
    handleClick() {
      console.log('Button clicked')
    }
  }
}
</script>
```

## Configuration Options

### Framework Detection

Homeostasis automatically detects Vue.js usage through:

- **File Extensions**: `.vue`, `.js`, `.ts`
- **Error Patterns**: Vue-specific error messages
- **Dependencies**: `vue`, `vuex`, `vue-router` in package.json
- **Framework Context**: Explicit framework configuration

### Error Analysis Settings

```javascript
const homeostasis = createHomeostasis({
  vue: {
    // Composition API settings
    compositionApi: {
      enforceImports: true,
      requireReturnFromSetup: true,
      validateReactivity: true
    },
    
    // Template validation
    template: {
      requireKeyInVFor: true,
      warnVIfWithVFor: true,
      validateDirectives: true
    },
    
    // Vuex settings
    vuex: {
      preventDirectMutation: true,
      requireMutationTypes: true,
      validateActions: true
    },
    
    // Router settings
    router: {
      requireGuardNext: true,
      validateRoutes: true,
      preventRedirectLoops: true
    }
  }
})
```

## Advanced Features

### 1. Vue 3 Specific Support

**Script Setup Syntax**
```vue
<!-- ✅ Supported: script setup syntax -->
<script setup>
import { ref, computed } from 'vue'

const count = ref(0)
const doubled = computed(() => count.value * 2)

function increment() {
  count.value++
}
</script>

<template>
  <div>
    <p>Count: {{ count }}</p>
    <p>Doubled: {{ doubled }}</p>
    <button @click="increment">Increment</button>
  </div>
</template>
```

**Teleport and Suspense**
```vue
<!-- ✅ Supported: Vue 3 features -->
<template>
  <div>
    <!-- Teleport to modal -->
    <Teleport to="#modal-target">
      <div class="modal">
        <p>Modal content</p>
      </div>
    </Teleport>
    
    <!-- Suspense for async components -->
    <Suspense>
      <template #default>
        <AsyncComponent />
      </template>
      <template #fallback>
        <div>Loading...</div>
      </template>
    </Suspense>
  </div>
</template>
```

### 2. Custom Directive Support

```javascript
// ✅ Supported: custom directive validation
app.directive('focus', {
  mounted(el) {
    el.focus()
  }
})
```

### 3. Provide/Inject Pattern

```javascript
// ✅ Supported: provide/inject validation
// Parent component
export default {
  setup() {
    provide('message', 'Hello from parent')
  }
}

// Child component
export default {
  setup() {
    const message = inject('message', 'Default message')
    return { message }
  }
}
```

## Build Tool Integration

### Vite Integration

```javascript
// vite.config.js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import homeostasis from 'homeostasis-vue/vite'

export default defineConfig({
  plugins: [
    vue(),
    homeostasis({
      framework: 'vue',
      features: ['composition-api', 'vuex', 'router'],
      development: true // Enable in development mode
    })
  ]
})
```

### Webpack Integration

```javascript
// webpack.config.js
const HomeostasisPlugin = require('homeostasis-vue/webpack')

module.exports = {
  plugins: [
    new HomeostasisPlugin({
      framework: 'vue',
      features: ['composition-api', 'vuex', 'router']
    })
  ]
}
```

## Performance Considerations

### Monitoring Overhead

- **Development**: Full monitoring enabled
- **Production**: Error reporting only
- **Build Time**: Template validation and optimization

### Memory Usage

```javascript
const homeostasis = createHomeostasis({
  performance: {
    maxCacheSize: 1000,
    cacheTimeout: 300000, // 5 minutes
    enableMetrics: process.env.NODE_ENV === 'development'
  }
})
```

## Best Practices

### 1. Component Organization

```javascript
// ✅ Good: Clear component structure
export default {
  name: 'UserProfile',
  props: {
    user: {
      type: Object,
      required: true
    }
  },
  emits: ['update', 'delete'],
  setup(props, { emit }) {
    // Composition API logic
    return {
      // Exposed properties
    }
  }
}
```

### 2. Error Handling

```javascript
// ✅ Good: Proper error handling
export default {
  setup() {
    const error = ref(null)
    
    const fetchData = async () => {
      try {
        const response = await api.getData()
        return response.data
      } catch (err) {
        error.value = err.message
        throw err
      }
    }
    
    return { error, fetchData }
  }
}
```

### 3. State Management

```javascript
// ✅ Good: Proper Vuex usage
const store = createStore({
  strict: process.env.NODE_ENV !== 'production',
  state: {
    user: null,
    loading: false
  },
  mutations: {
    SET_USER(state, user) {
      state.user = user
    },
    SET_LOADING(state, loading) {
      state.loading = loading
    }
  },
  actions: {
    async fetchUser({ commit }, userId) {
      commit('SET_LOADING', true)
      try {
        const user = await api.getUser(userId)
        commit('SET_USER', user)
      } finally {
        commit('SET_LOADING', false)
      }
    }
  }
})
```

## Troubleshooting

### Common Issues

1. **Plugin Not Loading**
   ```bash
   # Check if Vue plugin is registered
   npx homeostasis plugins list
   ```

2. **False Positives**
   ```javascript
   // Disable specific rules
   const homeostasis = createHomeostasis({
     rules: {
       'vue-missing-key': 'off',
       'vuex-direct-mutation': 'warn'
     }
   })
   ```

3. **Performance Issues**
   ```javascript
   // Reduce monitoring scope
   const homeostasis = createHomeostasis({
     include: ['src/**/*.vue'],
     exclude: ['node_modules/**', 'dist/**']
   })
   ```

### Debug Mode

```javascript
const homeostasis = createHomeostasis({
  debug: true,
  logLevel: 'debug'
})
```

## Migration Guides

### Vue 2 to Vue 3

Homeostasis can help identify and fix Vue 2 to Vue 3 migration issues:

- Global API changes (`new Vue()` → `createApp()`)
- Composition API migration
- Event API changes
- Filter removal
- Breaking changes in reactivity

### Options API to Composition API

Automated refactoring suggestions for migrating from Options API to Composition API:

```javascript
// Before: Options API
export default {
  data() {
    return {
      count: 0
    }
  },
  computed: {
    doubled() {
      return this.count * 2
    }
  },
  methods: {
    increment() {
      this.count++
    }
  }
}

// After: Composition API (auto-suggested)
export default {
  setup() {
    const count = ref(0)
    const doubled = computed(() => count.value * 2)
    
    const increment = () => {
      count.value++
    }
    
    return {
      count,
      doubled,
      increment
    }
  }
}
```

## Contributing

To contribute to Vue.js support in Homeostasis:

1. Fork the repository
2. Create feature branch: `git checkout -b vue-feature-name`
3. Add tests for new Vue error patterns
4. Update rule files in `modules/analysis/rules/vue/`
5. Submit pull request

## License

Vue.js integration is part of the Homeostasis project and follows the same license terms.