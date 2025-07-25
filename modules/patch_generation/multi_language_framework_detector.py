#!/usr/bin/env python3
"""
Multi-Language and Framework Detection for LLM Patch Generation

This module provides comprehensive language and framework detection capabilities
to enable LLM-based patch generation across multiple programming languages and frameworks.
"""

import json
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SCALA = "scala"
    ELIXIR = "elixir"
    CLOJURE = "clojure"
    CPP = "cpp"
    C = "c"
    # Additional languages from Phase 12.A
    ZIG = "zig"
    NIM = "nim"
    CRYSTAL = "crystal"
    HASKELL = "haskell"
    FSHARP = "fsharp"
    ERLANG = "erlang"
    SQL = "sql"
    BASH = "bash"
    POWERSHELL = "powershell"
    LUA = "lua"
    R = "r"
    MATLAB = "matlab"
    JULIA = "julia"
    TERRAFORM = "terraform"
    ANSIBLE = "ansible"
    YAML = "yaml"
    JSON = "json"
    DOCKERFILE = "dockerfile"
    UNKNOWN = "unknown"


@dataclass
class FrameworkInfo:
    """Information about a detected framework."""
    name: str
    language: LanguageType
    version: Optional[str] = None
    confidence: float = 0.0
    indicators: List[str] = None
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = []


@dataclass
class LanguageInfo:
    """Information about detected language and frameworks."""
    language: LanguageType
    confidence: float
    frameworks: List[FrameworkInfo]
    file_patterns: List[str]
    language_features: Dict[str, Any]


class MultiLanguageFrameworkDetector:
    """
    Comprehensive detector for programming languages and frameworks.
    
    This detector can identify:
    1. Programming language from file extensions and content
    2. Frameworks and libraries being used
    3. Language-specific features and patterns
    4. Project structure and configuration files
    """

    def __init__(self):
        """Initialize the detector with language and framework patterns."""
        self.language_patterns = self._init_language_patterns()
        self.framework_patterns = self._init_framework_patterns()
        self.file_extension_map = self._init_file_extension_map()
        
    def _init_language_patterns(self) -> Dict[LanguageType, Dict[str, Any]]:
        """Initialize language-specific detection patterns."""
        return {
            LanguageType.PYTHON: {
                'imports': [
                    r'import\s+\w+',
                    r'from\s+\w+\s+import',
                    r'import\s+\w+\.\w+',
                ],
                'syntax': [
                    r'def\s+\w+\s*\(',
                    r'class\s+\w+\s*\(',
                    r'if\s+__name__\s*==\s*["\']__main__["\']',
                    r'@\w+',  # decorators
                    r':\s*$',  # colon at end of line
                ],
                'keywords': [
                    'def', 'class', 'import', 'from', 'if', 'elif', 'else',
                    'try', 'except', 'finally', 'with', 'as', 'lambda'
                ],
                'comment_style': '#',
                'indent_style': 'spaces',
                'typical_indent': 4
            },
            LanguageType.JAVASCRIPT: {
                'imports': [
                    r'import\s+.*\s+from\s+["\'].+["\']',
                    r'const\s+.*=\s*require\s*\(',
                    r'import\s*\(',  # dynamic imports
                ],
                'syntax': [
                    r'function\s+\w+\s*\(',
                    r'const\s+\w+\s*=',
                    r'let\s+\w+\s*=',
                    r'var\s+\w+\s*=',
                    r'=>\s*{',  # arrow functions
                    r'}\s*;?\s*$',
                ],
                'keywords': [
                    'function', 'const', 'let', 'var', 'if', 'else', 'for',
                    'while', 'return', 'import', 'export', 'class', 'extends'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 2
            },
            LanguageType.TYPESCRIPT: {
                'imports': [
                    r'import\s+.*\s+from\s+["\'].+["\']',
                    r'import\s+type\s+',
                ],
                'syntax': [
                    r'interface\s+\w+\s*{',
                    r'type\s+\w+\s*=',
                    r':\s*\w+(\[\])?(\s*\|)?',  # type annotations
                    r'<\w+>',  # generics
                    r'as\s+\w+',  # type assertions
                ],
                'keywords': [
                    'interface', 'type', 'enum', 'namespace', 'declare',
                    'public', 'private', 'protected', 'readonly', 'abstract'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 2
            },
            LanguageType.JAVA: {
                'imports': [
                    r'import\s+[\w\.]+;',
                    r'package\s+[\w\.]+;',
                ],
                'syntax': [
                    r'public\s+class\s+\w+',
                    r'public\s+static\s+void\s+main',
                    r'@\w+',  # annotations
                    r'}\s*$',
                    r';\s*$',
                ],
                'keywords': [
                    'public', 'private', 'protected', 'static', 'final',
                    'class', 'interface', 'extends', 'implements', 'package'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 4
            },
            LanguageType.GO: {
                'imports': [
                    r'import\s+["\'][^"\']+["\']',
                    r'import\s+\(',
                    r'package\s+\w+',
                ],
                'syntax': [
                    r'func\s+\w+\s*\(',
                    r'type\s+\w+\s+struct',
                    r'type\s+\w+\s+interface',
                    r':=',  # short variable declaration
                    r'go\s+\w+\(',  # goroutines
                ],
                'keywords': [
                    'func', 'package', 'import', 'type', 'struct', 'interface',
                    'var', 'const', 'if', 'else', 'for', 'range', 'select', 'case'
                ],
                'comment_style': '//',
                'indent_style': 'tabs',
                'typical_indent': 1
            },
            LanguageType.RUST: {
                'imports': [
                    r'use\s+[\w:]+;',
                    r'extern\s+crate\s+\w+;',
                ],
                'syntax': [
                    r'fn\s+\w+\s*\(',
                    r'struct\s+\w+\s*{',
                    r'enum\s+\w+\s*{',
                    r'impl\s+\w+',
                    r'&\w+',  # references
                    r'->\s*\w+',  # return types
                ],
                'keywords': [
                    'fn', 'struct', 'enum', 'impl', 'trait', 'use', 'mod',
                    'let', 'mut', 'match', 'if', 'else', 'loop', 'while', 'for'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 4
            },
            LanguageType.SWIFT: {
                'imports': [
                    r'import\s+\w+',
                ],
                'syntax': [
                    r'func\s+\w+\s*\(',
                    r'class\s+\w+\s*:',
                    r'struct\s+\w+\s*{',
                    r'enum\s+\w+\s*{',
                    r'var\s+\w+\s*:',
                    r'let\s+\w+\s*=',
                ],
                'keywords': [
                    'func', 'class', 'struct', 'enum', 'protocol', 'extension',
                    'var', 'let', 'if', 'else', 'guard', 'switch', 'case', 'for'
                ],
                'comment_style': '//',
                'indent_style': 'spaces',
                'typical_indent': 4
            }
        }

    def _init_framework_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize framework detection patterns."""
        return {
            # Python Frameworks
            'django': {
                'language': LanguageType.PYTHON,
                'patterns': [
                    r'from\s+django',
                    r'import\s+django',
                    r'DJANGO_SETTINGS_MODULE',
                    r'django\.db\.',
                    r'models\.Model',
                    r'django\.contrib\.',
                    r'django\.urls',
                    r'django\.conf',
                    r'ModelAdmin',
                    r'path\s*\(',
                    r'include\s*\(',
                    r'django\.shortcuts',
                    r'django\.views',
                    r'django\.forms',
                    r'django\.test',
                ],
                'files': ['manage.py', 'settings.py', 'urls.py', 'wsgi.py', 'asgi.py'],
                'directories': ['migrations/', 'templates/', 'static/', 'media/'],
                'config_files': ['requirements.txt', 'Pipfile', 'setup.py', 'pyproject.toml']
            },
            'flask': {
                'language': LanguageType.PYTHON,
                'patterns': [
                    r'from\s+flask\s+import',
                    r'Flask\s*\(',
                    r'@app\.route',
                    r'flask\.Flask',
                    r'render_template\s*\(',
                    r'request\.',
                    r'jsonify\s*\(',
                    r'Blueprint\s*\(',
                    r'flask_sqlalchemy',
                    r'flask_migrate',
                    r'flask_login',
                    r'flask_wtf',
                    r'flask_cors',
                    r'make_response\s*\(',
                    r'redirect\s*\(',
                    r'url_for\s*\(',
                ],
                'files': ['app.py', 'run.py', 'application.py', 'wsgi.py'],
                'directories': ['templates/', 'static/'],
                'config_files': ['requirements.txt', 'Pipfile', 'pyproject.toml']
            },
            'fastapi': {
                'language': LanguageType.PYTHON,
                'patterns': [
                    r'from\s+fastapi\s+import',
                    r'FastAPI\s*\(',
                    r'@app\.(get|post|put|delete|patch|options|head)',
                    r'Depends\s*\(',
                    r'APIRouter\s*\(',
                    r'BaseModel',
                    r'from\s+pydantic',
                    r'HTTPException\s*\(',
                    r'status_code\s*=',
                    r'response_model\s*=',
                    r'Body\s*\(',
                    r'Query\s*\(',
                    r'Path\s*\(',
                    r'BackgroundTasks',
                    r'UploadFile',
                ],
                'files': ['main.py', 'app.py', 'api.py'],
                'directories': ['routers/', 'models/', 'schemas/'],
                'config_files': ['requirements.txt', 'Pipfile', 'pyproject.toml']
            },
            'pyramid': {
                'language': LanguageType.PYTHON,
                'patterns': [
                    r'from\s+pyramid',
                    r'import\s+pyramid',
                    r'pyramid\.config',
                    r'Configurator\s*\(',
                    r'config\.add_route',
                    r'config\.scan',
                    r'@view_config',
                    r'pyramid\.view',
                    r'pyramid\.response',
                    r'pyramid\.httpexceptions',
                    r'pyramid\.renderers',
                    r'pyramid\.security',
                    r'pyramid\.authentication',
                    r'pyramid\.authorization',
                ],
                'files': ['__init__.py', 'views.py', 'models.py', 'development.ini', 'production.ini'],
                'directories': ['templates/', 'static/', 'alembic/'],
                'config_files': ['requirements.txt', 'setup.py', 'setup.cfg', 'pyproject.toml']
            },
            'tornado': {
                'language': LanguageType.PYTHON,
                'patterns': [
                    r'import\s+tornado',
                    r'from\s+tornado',
                    r'tornado\.web',
                    r'tornado\.ioloop',
                    r'RequestHandler',
                    r'Application\s*\(',
                    r'tornado\.gen',
                    r'@tornado\.gen\.coroutine',
                    r'tornado\.httpserver',
                    r'tornado\.options',
                    r'tornado\.escape',
                    r'tornado\.websocket',
                    r'WebSocketHandler',
                    r'tornado\.template',
                    r'IOLoop\.current\(\)',
                ],
                'files': ['app.py', 'server.py', 'main.py'],
                'directories': ['templates/', 'static/', 'handlers/'],
                'config_files': ['requirements.txt', 'Pipfile', 'pyproject.toml']
            },
            
            # JavaScript/TypeScript Frameworks
            'react': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'import\s+React',
                    r'from\s+["\']react["\']',
                    r'useState\s*\(',
                    r'useEffect\s*\(',
                    r'className=',
                    r'JSX\.Element',
                    r'useContext\s*\(',
                    r'useReducer\s*\(',
                    r'useCallback\s*\(',
                    r'useMemo\s*\(',
                    r'useRef\s*\(',
                    r'useLayoutEffect\s*\(',
                    r'React\.Component',
                    r'React\.PureComponent',
                    r'React\.memo',
                    r'React\.Fragment',
                    r'ReactDOM\.render',
                    r'createRoot\s*\(',
                ],
                'files': ['App.js', 'App.jsx', 'App.tsx', 'index.js', 'index.jsx', 'index.tsx'],
                'directories': ['src/components/', 'src/pages/', 'src/hooks/'],
                'config_files': ['package.json', '.babelrc', 'webpack.config.js']
            },
            'redux': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']redux["\']',
                    r'from\s+["\']react-redux["\']',
                    r'createStore\s*\(',
                    r'combineReducers\s*\(',
                    r'applyMiddleware\s*\(',
                    r'useSelector\s*\(',
                    r'useDispatch\s*\(',
                    r'connect\s*\(',
                    r'Provider\s+.*store=',
                    r'dispatch\s*\(',
                    r'getState\s*\(',
                    r'createSlice\s*\(',
                    r'configureStore\s*\(',
                    r'@reduxjs/toolkit',
                ],
                'files': ['store.js', 'store.ts', 'redux/store.js'],
                'directories': ['redux/', 'store/', 'reducers/', 'actions/'],
                'config_files': ['package.json']
            },
            'mobx': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']mobx["\']',
                    r'from\s+["\']mobx-react["\']',
                    r'from\s+["\']mobx-react-lite["\']',
                    r'@observable',
                    r'@computed',
                    r'@action',
                    r'@observer',
                    r'makeObservable\s*\(',
                    r'makeAutoObservable\s*\(',
                    r'observable\s*\(',
                    r'action\s*\(',
                    r'computed\s*\(',
                    r'observer\s*\(',
                    r'runInAction\s*\(',
                ],
                'files': ['stores/', 'mobx/'],
                'directories': ['stores/', 'mobx/'],
                'config_files': ['package.json', '.babelrc']
            },
            'react-query': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']react-query["\']',
                    r'from\s+["\']@tanstack/react-query["\']',
                    r'useQuery\s*\(',
                    r'useMutation\s*\(',
                    r'useInfiniteQuery\s*\(',
                    r'useQueries\s*\(',
                    r'QueryClient',
                    r'QueryClientProvider',
                    r'queryClient',
                    r'invalidateQueries\s*\(',
                    r'prefetchQuery\s*\(',
                    r'setQueryData\s*\(',
                ],
                'files': [],
                'directories': ['queries/', 'api/'],
                'config_files': ['package.json']
            },
            'recoil': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']recoil["\']',
                    r'atom\s*\(',
                    r'selector\s*\(',
                    r'useRecoilState\s*\(',
                    r'useRecoilValue\s*\(',
                    r'useSetRecoilState\s*\(',
                    r'RecoilRoot',
                    r'atomFamily\s*\(',
                    r'selectorFamily\s*\(',
                    r'useRecoilCallback\s*\(',
                    r'waitForAll\s*\(',
                    r'waitForAny\s*\(',
                ],
                'files': [],
                'directories': ['atoms/', 'selectors/', 'recoil/'],
                'config_files': ['package.json']
            },
            'vue': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'import\s+Vue',
                    r'from\s+["\']vue["\']',
                    r'<template>',
                    r'<script>',
                    r'Vue\.component',
                    r'createApp\s*\(',
                    r'defineComponent\s*\(',
                    r'ref\s*\(',
                    r'reactive\s*\(',
                    r'computed\s*\(',
                    r'watch\s*\(',
                    r'onMounted\s*\(',
                    r'onBeforeMount\s*\(',
                    r'v-model',
                    r'v-if',
                    r'v-for',
                    r'v-show',
                    r'@click',
                    r':class',
                ],
                'files': ['App.vue', 'main.js', 'main.ts'],
                'directories': ['src/components/', 'src/views/', 'src/pages/'],
                'config_files': ['package.json', 'vue.config.js', 'vite.config.js']
            },
            'vuex': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']vuex["\']',
                    r'import\s+Vuex',
                    r'createStore\s*\(',
                    r'new\s+Vuex\.Store',
                    r'state:\s*\{',
                    r'mutations:\s*\{',
                    r'actions:\s*\{',
                    r'getters:\s*\{',
                    r'modules:\s*\{',
                    r'commit\s*\(',
                    r'dispatch\s*\(',
                    r'mapState\s*\(',
                    r'mapGetters\s*\(',
                    r'mapActions\s*\(',
                    r'mapMutations\s*\(',
                ],
                'files': ['store.js', 'store/index.js', 'store.ts'],
                'directories': ['store/', 'stores/'],
                'config_files': ['package.json']
            },
            'pinia': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']pinia["\']',
                    r'createPinia\s*\(',
                    r'defineStore\s*\(',
                    r'storeToRefs\s*\(',
                    r'useStore',
                    r'\$patch\s*\(',
                    r'\$reset\s*\(',
                    r'\$subscribe\s*\(',
                    r'acceptHMRUpdate\s*\(',
                ],
                'files': [],
                'directories': ['stores/', 'store/'],
                'config_files': ['package.json']
            },
            'nuxt': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']nuxt["\']',
                    r'from\s+["\']@nuxt/',
                    r'nuxt\.config',
                    r'defineNuxtConfig\s*\(',
                    r'useAsyncData\s*\(',
                    r'useFetch\s*\(',
                    r'useRoute\s*\(',
                    r'useRouter\s*\(',
                    r'useState\s*\(',
                    r'useNuxtData\s*\(',
                    r'navigateTo\s*\(',
                    r'\$fetch\s*\(',
                    r'pages/',
                    r'layouts/',
                ],
                'files': ['nuxt.config.js', 'nuxt.config.ts', 'app.vue'],
                'directories': ['pages/', 'layouts/', 'components/', 'composables/', 'server/'],
                'config_files': ['package.json', 'nuxt.config.js', 'nuxt.config.ts']
            },
            'vue-router': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']vue-router["\']',
                    r'createRouter\s*\(',
                    r'createWebHistory\s*\(',
                    r'createWebHashHistory\s*\(',
                    r'useRoute\s*\(',
                    r'useRouter\s*\(',
                    r'RouterView',
                    r'RouterLink',
                    r'router-view',
                    r'router-link',
                    r'routes:\s*\[',
                    r'path:\s*["\']',
                    r'beforeEach\s*\(',
                    r'beforeResolve\s*\(',
                    r'\$route',
                    r'\$router',
                ],
                'files': ['router.js', 'router/index.js', 'router.ts'],
                'directories': ['router/', 'routes/'],
                'config_files': ['package.json']
            },
            'angular': {
                'language': LanguageType.TYPESCRIPT,
                'patterns': [
                    r'@Component\s*\(',
                    r'@Injectable\s*\(',
                    r'@NgModule\s*\(',
                    r'from\s+["\']@angular/',
                    r'@Directive\s*\(',
                    r'@Pipe\s*\(',
                    r'@Input\s*\(',
                    r'@Output\s*\(',
                    r'@ViewChild\s*\(',
                    r'@ContentChild\s*\(',
                    r'@HostListener\s*\(',
                    r'@HostBinding\s*\(',
                    r'OnInit',
                    r'OnDestroy',
                    r'AfterViewInit',
                    r'HttpClient',
                    r'Observable',
                    r'Subject',
                    r'BehaviorSubject',
                    r'FormControl',
                    r'FormGroup',
                    r'FormBuilder',
                    r'RouterModule',
                    r'ActivatedRoute',
                ],
                'files': ['angular.json', 'main.ts', 'polyfills.ts'],
                'directories': ['src/app/', 'src/assets/', 'src/environments/'],
                'config_files': ['package.json', 'tsconfig.json', 'tsconfig.app.json']
            },
            'rxjs': {
                'language': LanguageType.TYPESCRIPT,
                'patterns': [
                    r'from\s+["\']rxjs["\']',
                    r'from\s+["\']rxjs/operators["\']',
                    r'Observable',
                    r'Subject',
                    r'BehaviorSubject',
                    r'ReplaySubject',
                    r'AsyncSubject',
                    r'of\s*\(',
                    r'from\s*\(',
                    r'pipe\s*\(',
                    r'map\s*\(',
                    r'filter\s*\(',
                    r'tap\s*\(',
                    r'catchError\s*\(',
                    r'switchMap\s*\(',
                    r'mergeMap\s*\(',
                    r'concatMap\s*\(',
                    r'exhaustMap\s*\(',
                    r'debounceTime\s*\(',
                    r'distinctUntilChanged\s*\(',
                    r'takeUntil\s*\(',
                    r'subscribe\s*\(',
                ],
                'files': [],
                'directories': [],
                'config_files': ['package.json']
            },
            'ngrx': {
                'language': LanguageType.TYPESCRIPT,
                'patterns': [
                    r'from\s+["\']@ngrx/store["\']',
                    r'from\s+["\']@ngrx/effects["\']',
                    r'from\s+["\']@ngrx/entity["\']',
                    r'from\s+["\']@ngrx/router-store["\']',
                    r'createAction\s*\(',
                    r'createReducer\s*\(',
                    r'createEffect\s*\(',
                    r'createSelector\s*\(',
                    r'createFeatureSelector\s*\(',
                    r'Store<',
                    r'Actions',
                    r'Effect',
                    r'ofType\s*\(',
                    r'dispatch\s*\(',
                    r'select\s*\(',
                    r'props\s*\(',
                    r'on\s*\(',
                    r'StoreModule',
                    r'EffectsModule',
                ],
                'files': [],
                'directories': ['store/', 'state/', 'ngrx/'],
                'config_files': ['package.json']
            },
            'angular-material': {
                'language': LanguageType.TYPESCRIPT,
                'patterns': [
                    r'from\s+["\']@angular/material',
                    r'from\s+["\']@angular/cdk',
                    r'MatDialog',
                    r'MatSnackBar',
                    r'MatTable',
                    r'MatPaginator',
                    r'MatSort',
                    r'MatFormField',
                    r'MatInput',
                    r'MatButton',
                    r'MatIcon',
                    r'MatToolbar',
                    r'MatSidenav',
                    r'MatCard',
                    r'MatSelect',
                    r'MatDatepicker',
                    r'mat-',
                    r'<mat-',
                ],
                'files': [],
                'directories': [],
                'config_files': ['package.json', 'angular.json']
            },
            'nextjs': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'from\s+["\']next/',
                    r'getStaticProps',
                    r'getServerSideProps',
                    r'pages/_app',
                ],
                'files': ['next.config.js'],
                'config_files': ['package.json']
            },
            'express': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'require\s*\(\s*["\']express["\']\s*\)',
                    r'import\s+express\s+from\s+["\']express["\']',
                    r'express\s*\(\s*\)',
                    r'app\.(get|post|put|delete|patch|use)\s*\(',
                    r'app\.listen\s*\(',
                    r'Router\s*\(\s*\)',
                    r'express\.Router',
                    r'express\.static',
                    r'express\.json',
                    r'express\.urlencoded',
                    r'req\s*,\s*res\s*,?\s*next?',
                    r'res\.(send|json|status|redirect)',
                    r'middleware',
                    r'body-parser',
                    r'morgan',
                ],
                'files': ['app.js', 'server.js', 'index.js'],
                'directories': ['routes/', 'middleware/', 'controllers/'],
                'config_files': ['package.json']
            },
            'koa': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'require\s*\(\s*["\']koa["\']\s*\)',
                    r'import\s+Koa\s+from\s+["\']koa["\']',
                    r'new\s+Koa\s*\(\s*\)',
                    r'app\.use\s*\(\s*async',
                    r'ctx\.(body|status|request|response)',
                    r'koa-router',
                    r'koa-bodyparser',
                    r'koa-static',
                    r'koa-compose',
                    r'ctx\.throw',
                    r'ctx\.assert',
                    r'await\s+next\s*\(\s*\)',
                ],
                'files': ['app.js', 'server.js', 'index.js'],
                'directories': ['middleware/', 'routes/'],
                'config_files': ['package.json']
            },
            'hapi': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'require\s*\(\s*["\']@hapi/hapi["\']\s*\)',
                    r'import\s+Hapi\s+from\s+["\']@hapi/hapi["\']',
                    r'Hapi\.server\s*\(',
                    r'server\.route\s*\(',
                    r'server\.start\s*\(',
                    r'server\.register\s*\(',
                    r'method:\s*["\']GET["\']',
                    r'handler:\s*\w+',
                    r'request\.payload',
                    r'h\.response',
                    r'@hapi/joi',
                    r'@hapi/boom',
                    r'server\.plugin',
                ],
                'files': ['server.js', 'index.js'],
                'directories': ['routes/', 'plugins/', 'handlers/'],
                'config_files': ['package.json']
            },
            'fastify': {
                'language': LanguageType.JAVASCRIPT,
                'patterns': [
                    r'require\s*\(\s*["\']fastify["\']\s*\)',
                    r'import\s+fastify\s+from\s+["\']fastify["\']',
                    r'fastify\s*\(\s*\{',
                    r'fastify\.(get|post|put|delete|patch|register)\s*\(',
                    r'fastify\.listen\s*\(',
                    r'reply\.(send|code|type|header)',
                    r'request\.(body|query|params|headers)',
                    r'fastify-plugin',
                    r'fastify-autoload',
                    r'fastify\.decorate',
                    r'fastify\.setErrorHandler',
                    r'fastify\.addHook',
                ],
                'files': ['app.js', 'server.js', 'index.js'],
                'directories': ['routes/', 'plugins/', 'schemas/'],
                'config_files': ['package.json']
            },
            'nestjs': {
                'language': LanguageType.TYPESCRIPT,
                'patterns': [
                    r'@nestjs/common',
                    r'@Module\s*\(',
                    r'@Controller\s*\(',
                    r'@Injectable\s*\(',
                    r'@Get\s*\(',
                    r'@Post\s*\(',
                    r'@Put\s*\(',
                    r'@Delete\s*\(',
                    r'@Patch\s*\(',
                    r'NestFactory',
                    r'@Body\s*\(',
                    r'@Param\s*\(',
                    r'@Query\s*\(',
                    r'@UseGuards\s*\(',
                    r'@UseInterceptors\s*\(',
                    r'providers:\s*\[',
                    r'controllers:\s*\[',
                    r'imports:\s*\[',
                ],
                'files': ['main.ts', 'app.module.ts'],
                'directories': ['src/', 'modules/', 'controllers/', 'services/'],
                'config_files': ['package.json', 'tsconfig.json', 'nest-cli.json']
            },
            'typeorm': {
                'language': LanguageType.TYPESCRIPT,
                'patterns': [
                    r'from\s+["\']typeorm["\']',
                    r'import\s+.*\s+from\s+["\']typeorm["\']',
                    r'@Entity\s*\(',
                    r'@Column\s*\(',
                    r'@PrimaryGeneratedColumn\s*\(',
                    r'@ManyToOne\s*\(',
                    r'@OneToMany\s*\(',
                    r'@ManyToMany\s*\(',
                    r'@OneToOne\s*\(',
                    r'@JoinTable\s*\(',
                    r'@JoinColumn\s*\(',
                    r'DataSource',
                    r'Repository',
                    r'QueryBuilder',
                    r'getRepository\s*\(',
                    r'createConnection\s*\(',
                    r'createQueryBuilder\s*\(',
                    r'@BeforeInsert\s*\(',
                    r'@AfterLoad\s*\(',
                ],
                'files': ['ormconfig.json', 'ormconfig.js', 'data-source.ts'],
                'directories': ['entities/', 'migrations/', 'repositories/'],
                'config_files': ['package.json', 'tsconfig.json']
            },
            'prisma': {
                'language': LanguageType.TYPESCRIPT,
                'patterns': [
                    r'@prisma/client',
                    r'PrismaClient',
                    r'new\s+PrismaClient\s*\(',
                    r'prisma\.\w+\.(create|update|delete|findMany|findUnique|findFirst)',
                    r'prisma\.\$connect',
                    r'prisma\.\$disconnect',
                    r'prisma\.\$transaction',
                    r'Prisma\.',
                    r'include:\s*{',
                    r'select:\s*{',
                    r'where:\s*{',
                    r'orderBy:\s*{',
                    r'@@map\s*\(',
                    r'@@id\s*\(',
                    r'@@unique\s*\(',
                ],
                'files': ['prisma/schema.prisma'],
                'directories': ['prisma/', 'generated/'],
                'config_files': ['package.json', 'tsconfig.json', '.env']
            },
            
            # Java Frameworks
            'spring': {
                'language': LanguageType.JAVA,
                'patterns': [
                    r'@SpringBootApplication',
                    r'@RestController',
                    r'@Service',
                    r'@Repository',
                    r'@Controller',
                    r'@Component',
                    r'@Configuration',
                    r'@Bean',
                    r'@Autowired',
                    r'@Value\s*\(',
                    r'@RequestMapping',
                    r'@GetMapping',
                    r'@PostMapping',
                    r'@PutMapping',
                    r'@DeleteMapping',
                    r'@PathVariable',
                    r'@RequestParam',
                    r'@RequestBody',
                    r'@ResponseBody',
                    r'@Transactional',
                    r'@EnableAutoConfiguration',
                    r'@ComponentScan',
                    r'@SpringBootTest',
                    r'@Test',
                    r'@MockBean',
                    r'@WebMvcTest',
                    r'@DataJpaTest',
                    r'@RestControllerAdvice',
                    r'@ExceptionHandler',
                    r'@Aspect',
                    r'@Before',
                    r'@After',
                    r'@Around',
                    r'@Pointcut',
                    r'org\.springframework\.',
                    r'import\s+org\.springframework\.',
                    r'ResponseEntity',
                    r'HttpStatus',
                    r'JpaRepository',
                    r'CrudRepository',
                    r'PagingAndSortingRepository',
                    r'@Query\s*\(',
                    r'@Modifying',
                    r'@EnableWebSecurity',
                    r'@EnableGlobalMethodSecurity',
                    r'SecurityFilterChain',
                    r'WebSecurityConfigurerAdapter',
                    r'@EnableScheduling',
                    r'@Scheduled\s*\(',
                    r'@EnableAsync',
                    r'@Async',
                    r'@EnableCaching',
                    r'@Cacheable',
                    r'@CacheEvict',
                    r'@CachePut',
                    r'@EnableEurekaClient',
                    r'@EnableDiscoveryClient',
                    r'@FeignClient',
                    r'@LoadBalanced',
                    r'RestTemplate',
                    r'WebClient',
                    r'ApplicationContext',
                    r'ApplicationRunner',
                    r'CommandLineRunner',
                ],
                'files': [
                    'application.properties', 
                    'application.yml', 
                    'application.yaml',
                    'bootstrap.properties',
                    'bootstrap.yml',
                    'application-dev.properties',
                    'application-prod.properties',
                    'application-test.properties'
                ],
                'directories': ['src/main/java/', 'src/test/java/', 'src/main/resources/'],
                'config_files': ['pom.xml', 'build.gradle', 'build.gradle.kts']
            },
            'spring-boot': {
                'language': LanguageType.JAVA,
                'patterns': [
                    r'@SpringBootApplication',
                    r'spring-boot-starter',
                    r'org\.springframework\.boot',
                    r'SpringApplication\.run',
                    r'@EnableAutoConfiguration',
                    r'@ConditionalOn',
                    r'@ConfigurationProperties',
                    r'spring\.boot\.',
                    r'SpringBootServletInitializer',
                    r'@SpringBootTest',
                    r'@DataJpaTest',
                    r'@WebMvcTest',
                    r'@JsonTest',
                    r'@RestClientTest',
                    r'@AutoConfigureMockMvc',
                    r'@MockBean',
                    r'@SpyBean',
                    r'TestRestTemplate',
                    r'WebTestClient',
                ],
                'files': [
                    'application.properties',
                    'application.yml',
                    'application.yaml',
                    'application-*.properties',
                    'application-*.yml'
                ],
                'directories': ['src/main/java/', 'src/test/java/'],
                'config_files': ['pom.xml', 'build.gradle', 'build.gradle.kts']
            },
            'hibernate': {
                'language': LanguageType.JAVA,
                'patterns': [
                    r'@Entity',
                    r'@Table\s*\(',
                    r'@Id',
                    r'@GeneratedValue',
                    r'@Column\s*\(',
                    r'@OneToMany',
                    r'@ManyToOne',
                    r'@OneToOne',
                    r'@ManyToMany',
                    r'@JoinColumn',
                    r'@JoinTable',
                    r'@Embeddable',
                    r'@Embedded',
                    r'@EmbeddedId',
                    r'@Inheritance',
                    r'@DiscriminatorColumn',
                    r'@DiscriminatorValue',
                    r'@NamedQuery',
                    r'@NamedQuery',
                    r'@SqlResultSetMapping',
                    r'@Temporal',
                    r'@Lob',
                    r'@Transient',
                    r'@Version',
                    r'@PrePersist',
                    r'@PostPersist',
                    r'@PreUpdate',
                    r'@PostUpdate',
                    r'@PreRemove',
                    r'@PostRemove',
                    r'org\.hibernate\.',
                    r'javax\.persistence\.',
                    r'jakarta\.persistence\.',
                    r'SessionFactory',
                    r'Session',
                    r'Transaction',
                    r'Criteria',
                    r'CriteriaBuilder',
                    r'CriteriaQuery',
                    r'EntityManager',
                    r'EntityManagerFactory',
                    r'TypedQuery',
                    r'Query',
                    r'HQL',
                    r'hibernate\.cfg\.xml',
                    r'hibernate\.properties',
                ],
                'files': [
                    'hibernate.cfg.xml',
                    'hibernate.properties',
                    'persistence.xml'
                ],
                'directories': ['src/main/java/', 'src/main/resources/'],
                'config_files': ['pom.xml', 'build.gradle', 'build.gradle.kts']
            },
            'struts': {
                'language': LanguageType.JAVA,
                'patterns': [
                    r'org\.apache\.struts',
                    r'struts\.xml',
                    r'struts-config\.xml',
                    r'ActionSupport',
                    r'ActionMapping',
                    r'ActionForm',
                    r'ActionForward',
                    r'ActionServlet',
                    r'@Action\s*\(',
                    r'@Result\s*\(',
                    r'@Results\s*\(',
                    r'@Namespace',
                    r'@ParentPackage',
                    r'@InterceptorRef',
                    r'@Before',
                    r'@After',
                    r'@BeforeResult',
                    r'@SkipValidation',
                    r'@Validations',
                    r'@RequiredFieldValidator',
                    r'@RequiredStringValidator',
                    r'@IntRangeFieldValidator',
                    r'@EmailValidator',
                    r'execute\s*\(\s*\)\s*{',
                    r'validate\s*\(\s*\)\s*{',
                    r'prepare\s*\(\s*\)\s*{',
                    r'ServletActionContext',
                    r'ActionContext',
                    r'ValueStack',
                    r'OGNL',
                ],
                'files': [
                    'struts.xml',
                    'struts-config.xml',
                    'struts.properties',
                    'validation.xml'
                ],
                'directories': ['src/main/java/', 'src/main/webapp/WEB-INF/'],
                'config_files': ['pom.xml', 'build.gradle', 'web.xml']
            },
            'play': {
                'language': LanguageType.JAVA,
                'patterns': [
                    r'play\.mvc\.',
                    r'play\.api\.',
                    r'play\.libs\.',
                    r'play\.data\.',
                    r'play\.db\.',
                    r'play\.cache\.',
                    r'play\.filters\.',
                    r'play\.inject\.',
                    r'Controller',
                    r'Result',
                    r'Action',
                    r'@Inject',
                    r'@Singleton',
                    r'play\.Logger',
                    r'Http\.Request',
                    r'Http\.Response',
                    r'Http\.Context',
                    r'Http\.Session',
                    r'Http\.Flash',
                    r'Http\.Cookie',
                    r'Form<',
                    r'DynamicForm',
                    r'routes\.',
                    r'reverse\.',
                    r'@BodyParser',
                    r'@Cached',
                    r'@Security\.Authenticated',
                    r'Promise<',
                    r'CompletionStage<',
                    r'WSClient',
                    r'WSRequest',
                    r'WSResponse',
                    r'Akka',
                    r'ActorSystem',
                    r'application\.conf',
                    r'routes',
                ],
                'files': [
                    'application.conf',
                    'routes',
                    'messages',
                    'logback.xml',
                    'build.sbt'
                ],
                'directories': [
                    'app/',
                    'app/controllers/',
                    'app/models/',
                    'app/views/',
                    'conf/',
                    'public/',
                    'test/'
                ],
                'config_files': ['build.sbt', 'project/build.properties', 'project/plugins.sbt']
            },
            
            # C++ Frameworks
            'qt': {
                'language': LanguageType.CPP,
                'patterns': [
                    r'#include\s*<Q',
                    r'QApplication',
                    r'QWidget',
                    r'QMainWindow',
                    r'QPushButton',
                    r'QLabel',
                    r'QString',
                    r'QObject',
                    r'QThread',
                    r'QTimer',
                    r'QFile',
                    r'QDir',
                    r'QSettings',
                    r'QNetworkAccessManager',
                    r'QNetworkRequest',
                    r'QNetworkReply',
                    r'QJsonDocument',
                    r'QJsonObject',
                    r'QJsonArray',
                    r'QDebug',
                    r'qDebug\s*\(\)',
                    r'Q_OBJECT',
                    r'Q_PROPERTY',
                    r'Q_INVOKABLE',
                    r'Q_SIGNAL',
                    r'Q_SLOT',
                    r'Q_ENUM',
                    r'Q_FLAG',
                    r'Q_DECLARE_METATYPE',
                    r'signals:',
                    r'slots:',
                    r'emit\s+',
                    r'connect\s*\(',
                    r'disconnect\s*\(',
                    r'QML',
                    r'QtQuick',
                    r'Qt::',
                    r'QT_VERSION',
                    r'MOC_',
                    r'moc_',
                    r'\.pro',
                    r'\.pri',
                    r'\.qml',
                    r'\.ui',
                ],
                'files': [
                    'CMakeLists.txt',
                    '*.pro',
                    '*.pri',
                    '*.qml',
                    '*.ui',
                    'main.cpp',
                    'mainwindow.cpp',
                    'mainwindow.h'
                ],
                'directories': ['src/', 'include/', 'ui/', 'qml/', 'resources/'],
                'config_files': ['CMakeLists.txt', '*.pro', '*.cmake', 'conanfile.txt']
            },
            'boost': {
                'language': LanguageType.CPP,
                'patterns': [
                    r'#include\s*<boost/',
                    r'boost::',
                    r'namespace\s+boost',
                    r'boost::shared_ptr',
                    r'boost::unique_ptr',
                    r'boost::weak_ptr',
                    r'boost::make_shared',
                    r'boost::function',
                    r'boost::bind',
                    r'boost::ref',
                    r'boost::cref',
                    r'boost::thread',
                    r'boost::mutex',
                    r'boost::lock_guard',
                    r'boost::unique_lock',
                    r'boost::condition_variable',
                    r'boost::asio',
                    r'boost::filesystem',
                    r'boost::regex',
                    r'boost::algorithm',
                    r'boost::format',
                    r'boost::lexical_cast',
                    r'boost::any',
                    r'boost::optional',
                    r'boost::variant',
                    r'boost::signals2',
                    r'boost::property_tree',
                    r'boost::program_options',
                    r'boost::date_time',
                    r'boost::chrono',
                    r'boost::random',
                    r'boost::uuid',
                    r'boost::spirit',
                    r'boost::fusion',
                    r'boost::mpl',
                    r'boost::type_traits',
                    r'boost::enable_if',
                    r'boost::is_same',
                    r'BOOST_',
                    r'BOOST_VERSION',
                    r'BOOST_ASSERT',
                    r'BOOST_STATIC_ASSERT',
                    r'BOOST_FOREACH',
                    r'BOOST_AUTO',
                ],
                'files': ['CMakeLists.txt', 'Jamfile', 'boost-build.jam'],
                'directories': ['src/', 'include/', 'lib/', 'boost/'],
                'config_files': ['CMakeLists.txt', 'conanfile.txt', 'vcpkg.json']
            },
            'poco': {
                'language': LanguageType.CPP,
                'patterns': [
                    r'#include\s*[<"]Poco/',
                    r'Poco::',
                    r'namespace\s+Poco',
                    r'Poco::Net::',
                    r'Poco::Util::',
                    r'Poco::XML::',
                    r'Poco::JSON::',
                    r'Poco::Data::',
                    r'Poco::Crypto::',
                    r'HTTPServer',
                    r'HTTPRequestHandler',
                    r'HTTPServerRequest',
                    r'HTTPServerResponse',
                    r'HTTPClientSession',
                    r'TCPServer',
                    r'TCPServerConnection',
                    r'Socket',
                    r'ServerSocket',
                    r'StreamSocket',
                    r'DatagramSocket',
                    r'Application',
                    r'ServerApplication',
                    r'Poco::Logger',
                    r'Poco::AutoPtr',
                    r'Poco::SharedPtr',
                    r'Poco::Thread',
                    r'Poco::Runnable',
                    r'Poco::Task',
                    r'Poco::TaskManager',
                    r'Poco::NotificationCenter',
                    r'Poco::Observer',
                    r'Poco::Timer',
                    r'Poco::File',
                    r'Poco::Path',
                    r'Poco::DirectoryIterator',
                    r'Poco::RegularExpression',
                    r'Poco::DateTime',
                    r'Poco::Timestamp',
                    r'Poco::Timespan',
                    r'POCO_',
                ],
                'files': ['CMakeLists.txt', 'Makefile'],
                'directories': ['src/', 'include/', 'lib/'],
                'config_files': ['CMakeLists.txt', 'conanfile.txt', 'vcpkg.json']
            },
            'stl': {
                'language': LanguageType.CPP,
                'patterns': [
                    r'#include\s*<algorithm>',
                    r'#include\s*<vector>',
                    r'#include\s*<string>',
                    r'#include\s*<map>',
                    r'#include\s*<set>',
                    r'#include\s*<unordered_map>',
                    r'#include\s*<unordered_set>',
                    r'#include\s*<queue>',
                    r'#include\s*<stack>',
                    r'#include\s*<deque>',
                    r'#include\s*<list>',
                    r'#include\s*<array>',
                    r'#include\s*<memory>',
                    r'#include\s*<functional>',
                    r'#include\s*<thread>',
                    r'#include\s*<mutex>',
                    r'#include\s*<condition_variable>',
                    r'#include\s*<future>',
                    r'#include\s*<chrono>',
                    r'#include\s*<random>',
                    r'#include\s*<regex>',
                    r'#include\s*<filesystem>',
                    r'#include\s*<optional>',
                    r'#include\s*<variant>',
                    r'#include\s*<any>',
                    r'std::vector',
                    r'std::string',
                    r'std::map',
                    r'std::unordered_map',
                    r'std::unique_ptr',
                    r'std::shared_ptr',
                    r'std::make_unique',
                    r'std::make_shared',
                    r'std::move',
                    r'std::forward',
                    r'std::thread',
                    r'std::async',
                    r'std::promise',
                    r'std::future',
                    r'std::function',
                    r'std::bind',
                    r'std::placeholders',
                    r'std::chrono',
                    r'std::regex',
                    r'std::optional',
                    r'std::variant',
                    r'std::visit',
                    r'std::get',
                    r'std::holds_alternative',
                    r'std::filesystem',
                    r'namespace\s+std',
                    r'using\s+namespace\s+std',
                ],
                'files': ['*.cpp', '*.hpp', '*.cc', '*.cxx', '*.h'],
                'directories': ['src/', 'include/', 'test/'],
                'config_files': ['CMakeLists.txt', 'Makefile', '.clang-format']
            },
            
            # Go Frameworks
            'gin': {
                'language': LanguageType.GO,
                'patterns': [
                    r'gin\.Default\s*\(',
                    r'gin\.New\s*\(',
                    r'c\s*\*gin\.Context',
                    r'github\.com/gin-gonic/gin',
                    r'gin\.Engine',
                    r'gin\.RouterGroup',
                    r'gin\.HandlerFunc',
                    r'gin\.H\{',
                    r'c\.JSON\s*\(',
                    r'c\.String\s*\(',
                    r'c\.HTML\s*\(',
                    r'c\.XML\s*\(',
                    r'c\.YAML\s*\(',
                    r'c\.ProtoBuf\s*\(',
                    r'c\.Redirect\s*\(',
                    r'c\.Abort\s*\(',
                    r'c\.AbortWithStatus\s*\(',
                    r'c\.AbortWithStatusJSON\s*\(',
                    r'c\.AbortWithError\s*\(',
                    r'c\.Bind\s*\(',
                    r'c\.ShouldBind\s*\(',
                    r'c\.ShouldBindJSON\s*\(',
                    r'c\.ShouldBindXML\s*\(',
                    r'c\.ShouldBindQuery\s*\(',
                    r'c\.Param\s*\(',
                    r'c\.Query\s*\(',
                    r'c\.PostForm\s*\(',
                    r'c\.Cookie\s*\(',
                    r'c\.SetCookie\s*\(',
                    r'gin\.Recovery\s*\(',
                    r'gin\.Logger\s*\(',
                    r'gin\.BasicAuth\s*\(',
                    r'router\.GET\s*\(',
                    r'router\.POST\s*\(',
                    r'router\.PUT\s*\(',
                    r'router\.DELETE\s*\(',
                    r'router\.PATCH\s*\(',
                    r'router\.HEAD\s*\(',
                    r'router\.OPTIONS\s*\(',
                    r'router\.Group\s*\(',
                    r'router\.Use\s*\(',
                    r'router\.Static\s*\(',
                    r'router\.StaticFile\s*\(',
                    r'router\.StaticFS\s*\(',
                    r'router\.LoadHTMLGlob\s*\(',
                    r'router\.LoadHTMLFiles\s*\(',
                ],
                'files': ['main.go', 'server.go', 'router.go', 'handlers.go'],
                'directories': ['handlers/', 'middleware/', 'routes/', 'controllers/'],
                'config_files': ['go.mod', 'go.sum', 'Makefile', 'Dockerfile']
            },
            'echo': {
                'language': LanguageType.GO,
                'patterns': [
                    r'echo\.New\s*\(',
                    r'e\s*\*echo\.Echo',
                    r'c\s*echo\.Context',
                    r'github\.com/labstack/echo',
                    r'echo\.HandlerFunc',
                    r'echo\.MiddlewareFunc',
                    r'e\.GET\s*\(',
                    r'e\.POST\s*\(',
                    r'e\.PUT\s*\(',
                    r'e\.DELETE\s*\(',
                    r'e\.PATCH\s*\(',
                    r'e\.HEAD\s*\(',
                    r'e\.OPTIONS\s*\(',
                    r'e\.Group\s*\(',
                    r'e\.Use\s*\(',
                    r'e\.Static\s*\(',
                    r'e\.File\s*\(',
                    r'e\.Start\s*\(',
                    r'e\.Logger',
                    r'e\.Debug',
                    r'e\.Pre\s*\(',
                    r'c\.JSON\s*\(',
                    r'c\.String\s*\(',
                    r'c\.HTML\s*\(',
                    r'c\.XML\s*\(',
                    r'c\.Blob\s*\(',
                    r'c\.Stream\s*\(',
                    r'c\.File\s*\(',
                    r'c\.Attachment\s*\(',
                    r'c\.Inline\s*\(',
                    r'c\.NoContent\s*\(',
                    r'c\.Redirect\s*\(',
                    r'c\.Error\s*\(',
                    r'c\.Param\s*\(',
                    r'c\.QueryParam\s*\(',
                    r'c\.FormValue\s*\(',
                    r'c\.Cookie\s*\(',
                    r'c\.SetCookie\s*\(',
                    r'c\.Bind\s*\(',
                    r'c\.Validate\s*\(',
                    r'c\.Request\s*\(',
                    r'c\.Response\s*\(',
                    r'echo\.WrapHandler\s*\(',
                    r'echo\.WrapMiddleware\s*\(',
                    r'middleware\.Logger\s*\(',
                    r'middleware\.Recover\s*\(',
                    r'middleware\.CORS\s*\(',
                    r'middleware\.JWT\s*\(',
                    r'middleware\.BasicAuth\s*\(',
                    r'middleware\.RateLimiter\s*\(',
                ],
                'files': ['main.go', 'server.go', 'routes.go', 'handlers.go'],
                'directories': ['handlers/', 'middleware/', 'routes/', 'api/'],
                'config_files': ['go.mod', 'go.sum', 'Makefile', 'Dockerfile']
            },
            'fiber': {
                'language': LanguageType.GO,
                'patterns': [
                    r'fiber\.New\s*\(',
                    r'app\s*\*fiber\.App',
                    r'c\s*\*fiber\.Ctx',
                    r'github\.com/gofiber/fiber',
                    r'fiber\.Config',
                    r'fiber\.Handler',
                    r'fiber\.Router',
                    r'app\.Get\s*\(',
                    r'app\.Post\s*\(',
                    r'app\.Put\s*\(',
                    r'app\.Delete\s*\(',
                    r'app\.Patch\s*\(',
                    r'app\.Head\s*\(',
                    r'app\.Options\s*\(',
                    r'app\.Group\s*\(',
                    r'app\.Use\s*\(',
                    r'app\.Static\s*\(',
                    r'app\.Mount\s*\(',
                    r'app\.Listen\s*\(',
                    r'c\.JSON\s*\(',
                    r'c\.Send\s*\(',
                    r'c\.SendString\s*\(',
                    r'c\.SendStatus\s*\(',
                    r'c\.SendFile\s*\(',
                    r'c\.Download\s*\(',
                    r'c\.Render\s*\(',
                    r'c\.Redirect\s*\(',
                    r'c\.Status\s*\(',
                    r'c\.Next\s*\(',
                    r'c\.Params\s*\(',
                    r'c\.Query\s*\(',
                    r'c\.Body\s*\(',
                    r'c\.BodyParser\s*\(',
                    r'c\.FormValue\s*\(',
                    r'c\.FormFile\s*\(',
                    r'c\.Cookie\s*\(',
                    r'c\.Cookies\s*\(',
                    r'c\.Locals\s*\(',
                    r'c\.SaveFile\s*\(',
                    r'middleware\.Logger\s*\(',
                    r'middleware\.Recover\s*\(',
                    r'middleware\.CORS\s*\(',
                    r'middleware\.Compress\s*\(',
                    r'middleware\.Cache\s*\(',
                    r'middleware\.Limiter\s*\(',
                ],
                'files': ['main.go', 'app.go', 'routes.go', 'handlers.go'],
                'directories': ['handlers/', 'middleware/', 'routes/', 'controllers/'],
                'config_files': ['go.mod', 'go.sum', 'fiber.toml', 'Makefile']
            },
            'beego': {
                'language': LanguageType.GO,
                'patterns': [
                    r'beego\.Run\s*\(',
                    r'beego\.Controller',
                    r'beego\.Router\s*\(',
                    r'github\.com/beego/beego',
                    r'github\.com/astaxie/beego',
                    r'beego\.AppConfig',
                    r'beego\.NewApp\s*\(',
                    r'beego\.BConfig',
                    r'beego\.Get\s*\(',
                    r'beego\.Post\s*\(',
                    r'beego\.Put\s*\(',
                    r'beego\.Delete\s*\(',
                    r'beego\.Head\s*\(',
                    r'beego\.Options\s*\(',
                    r'beego\.Any\s*\(',
                    r'c\.Ctx\.Input',
                    r'c\.Ctx\.Output',
                    r'c\.Ctx\.Request',
                    r'c\.Ctx\.ResponseWriter',
                    r'c\.Data\[',
                    r'c\.TplName',
                    r'c\.ServeJSON\s*\(',
                    r'c\.ServeXML\s*\(',
                    r'c\.ServeYAML\s*\(',
                    r'c\.Redirect\s*\(',
                    r'c\.Abort\s*\(',
                    r'c\.StopRun\s*\(',
                    r'c\.GetString\s*\(',
                    r'c\.GetInt\s*\(',
                    r'c\.GetBool\s*\(',
                    r'c\.GetFloat\s*\(',
                    r'c\.GetFile\s*\(',
                    r'c\.SaveToFile\s*\(',
                    r'c\.SetSession\s*\(',
                    r'c\.GetSession\s*\(',
                    r'orm\.RegisterModel\s*\(',
                    r'orm\.RegisterDataBase\s*\(',
                    r'orm\.RunSyncdb\s*\(',
                    r'beego\.NSNamespace\s*\(',
                    r'beego\.NSRouter\s*\(',
                    r'beego\.NSInclude\s*\(',
                    r'beego\.Filter',
                    r'beego\.InsertFilter\s*\(',
                ],
                'files': ['main.go', 'app.conf', 'routers/router.go'],
                'directories': ['controllers/', 'models/', 'routers/', 'views/', 'conf/', 'static/'],
                'config_files': ['go.mod', 'go.sum', 'conf/app.conf', 'bee.json']
            },
            'revel': {
                'language': LanguageType.GO,
                'patterns': [
                    r'github\.com/revel/revel',
                    r'revel\.Controller',
                    r'revel\.Result',
                    r'revel\.Router',
                    r'revel\.Filter',
                    r'revel\.OnAppStart\s*\(',
                    r'revel\.InterceptMethod\s*\(',
                    r'revel\.InterceptFunc\s*\(',
                    r'c\.Render\s*\(',
                    r'c\.RenderJSON\s*\(',
                    r'c\.RenderXML\s*\(',
                    r'c\.RenderText\s*\(',
                    r'c\.RenderTemplate\s*\(',
                    r'c\.RenderFile\s*\(',
                    r'c\.RenderError\s*\(',
                    r'c\.Redirect\s*\(',
                    r'c\.NotFound\s*\(',
                    r'c\.Todo\s*\(',
                    r'c\.Forbidden\s*\(',
                    r'c\.Params',
                    r'c\.Session',
                    r'c\.Flash',
                    r'c\.Validation',
                    r'c\.Request',
                    r'c\.Response',
                    r'revel\.Filters',
                    r'revel\.PanicFilter',
                    r'revel\.RouterFilter',
                    r'revel\.ParamsFilter',
                    r'revel\.SessionFilter',
                    r'revel\.FlashFilter',
                    r'revel\.ValidationFilter',
                    r'revel\.InterceptorFilter',
                    r'revel\.CompressFilter',
                    r'revel\.ActionInvoker',
                    r'revel\.Config',
                    r'revel\.RunMode',
                    r'revel\.DevMode',
                    r'revel\.ProdMode',
                    r'routes\.',
                ],
                'files': ['app/init.go', 'conf/app.conf', 'conf/routes'],
                'directories': ['app/', 'app/controllers/', 'app/models/', 'app/views/', 'conf/', 'public/', 'tests/'],
                'config_files': ['go.mod', 'go.sum', 'conf/app.conf', 'conf/routes']
            },
            
            # Rust Frameworks
            'actix': {
                'language': LanguageType.RUST,
                'patterns': [
                    r'actix_web::',
                    r'actix_web::\{',
                    r'use\s+actix_web',
                    r'HttpServer::new',
                    r'App::new',
                    r'actix-web\s*=',
                    r'#\[actix_web::main\]',
                    r'#\[actix::main\]',
                    r'web::Data',
                    r'web::Json',
                    r'web::Query',
                    r'web::Path',
                    r'web::Form',
                    r'web::Bytes',
                    r'web::resource',
                    r'web::route',
                    r'web::get',
                    r'web::post',
                    r'web::put',
                    r'web::delete',
                    r'web::patch',
                    r'web::head',
                    r'HttpRequest',
                    r'HttpResponse',
                    r'Responder',
                    r'#\[get\(',
                    r'#\[post\(',
                    r'#\[put\(',
                    r'#\[delete\(',
                    r'#\[patch\(',
                    r'#\[route\(',
                    r'middleware::',
                    r'middleware::Logger',
                    r'middleware::Compress',
                    r'middleware::DefaultHeaders',
                    r'middleware::ErrorHandlers',
                    r'service\(',
                    r'to\(',
                    r'route\(',
                    r'bind\(',
                    r'run\(\)',
                    r'await\?',
                    r'Ok\(HttpResponse::',
                    r'Result<HttpResponse',
                    r'actix_rt',
                    r'actix_cors',
                    r'actix_session',
                    r'actix_identity',
                    r'actix_files',
                    r'guard::',
                    r'Handler',
                    r'FromRequest',
                    r'ResponseError',
                ],
                'files': ['main.rs', 'lib.rs', 'server.rs', 'handlers.rs'],
                'directories': ['src/', 'src/handlers/', 'src/models/', 'src/middleware/'],
                'config_files': ['Cargo.toml', 'Cargo.lock', '.env']
            },
            'rocket': {
                'language': LanguageType.RUST,
                'patterns': [
                    r'#\[macro_use\]\s*extern\s+crate\s+rocket',
                    r'use\s+rocket',
                    r'rocket::',
                    r'#\[launch\]',
                    r'rocket\s*=',
                    r'#\[rocket::main\]',
                    r'#\[get\(',
                    r'#\[post\(',
                    r'#\[put\(',
                    r'#\[delete\(',
                    r'#\[patch\(',
                    r'#\[head\(',
                    r'#\[options\(',
                    r'#\[route\(',
                    r'#\[catch\(',
                    r'rocket::routes!',
                    r'rocket::catchers!',
                    r'rocket::build\(\)',
                    r'rocket::ignite\(\)',
                    r'mount\(',
                    r'register\(',
                    r'attach\(',
                    r'manage\(',
                    r'launch\(\)',
                    r'State<',
                    r'Json<',
                    r'Form<',
                    r'Query<',
                    r'Path<',
                    r'Data<',
                    r'&State<',
                    r'Result<',
                    r'Status',
                    r'status::',
                    r'response::',
                    r'request::',
                    r'FromForm',
                    r'FromFormField',
                    r'FromParam',
                    r'FromSegments',
                    r'FromRequest',
                    r'Responder',
                    r'Handler',
                    r'Fairing',
                    r'fairings::',
                    r'Shield',
                    r'Cors',
                    r'Template',
                    r'NamedFile',
                    r'Redirect',
                    r'Flash',
                    r'Cookies',
                    r'CookieJar',
                    r'rocket_contrib',
                    r'rocket_cors',
                    r'rocket_db_pools',
                ],
                'files': ['main.rs', 'lib.rs', 'routes.rs'],
                'directories': ['src/', 'src/routes/', 'src/models/', 'src/guards/'],
                'config_files': ['Cargo.toml', 'Cargo.lock', 'Rocket.toml']
            },
            'tokio': {
                'language': LanguageType.RUST,
                'patterns': [
                    r'tokio::',
                    r'use\s+tokio',
                    r'tokio\s*=',
                    r'#\[tokio::main\]',
                    r'#\[tokio::test\]',
                    r'tokio::spawn',
                    r'tokio::select!',
                    r'tokio::join!',
                    r'tokio::try_join!',
                    r'tokio::pin!',
                    r'tokio::time::',
                    r'tokio::time::sleep',
                    r'tokio::time::timeout',
                    r'tokio::time::interval',
                    r'tokio::time::Duration',
                    r'tokio::time::Instant',
                    r'tokio::task::',
                    r'tokio::task::spawn',
                    r'tokio::task::spawn_blocking',
                    r'tokio::task::yield_now',
                    r'tokio::task::JoinHandle',
                    r'tokio::sync::',
                    r'tokio::sync::Mutex',
                    r'tokio::sync::RwLock',
                    r'tokio::sync::Semaphore',
                    r'tokio::sync::mpsc',
                    r'tokio::sync::oneshot',
                    r'tokio::sync::broadcast',
                    r'tokio::sync::watch',
                    r'tokio::sync::Notify',
                    r'tokio::sync::Barrier',
                    r'tokio::io::',
                    r'tokio::io::AsyncRead',
                    r'tokio::io::AsyncWrite',
                    r'tokio::io::AsyncBufRead',
                    r'tokio::io::AsyncSeek',
                    r'tokio::net::',
                    r'tokio::net::TcpListener',
                    r'tokio::net::TcpStream',
                    r'tokio::net::UdpSocket',
                    r'tokio::net::UnixListener',
                    r'tokio::net::UnixStream',
                    r'tokio::fs::',
                    r'tokio::process::',
                    r'tokio::signal::',
                    r'tokio::runtime::',
                    r'Runtime::new',
                    r'Builder::new',
                    r'\.await',
                    r'async\s+fn',
                    r'async\s+move',
                    r'async\s+\{',
                ],
                'files': ['main.rs', 'lib.rs'],
                'directories': ['src/', 'src/bin/'],
                'config_files': ['Cargo.toml', 'Cargo.lock']
            },
            'diesel': {
                'language': LanguageType.RUST,
                'patterns': [
                    r'diesel::',
                    r'use\s+diesel',
                    r'diesel\s*=',
                    r'#\[derive\(Queryable',
                    r'#\[derive\(Insertable',
                    r'#\[derive\(AsChangeset',
                    r'#\[derive\(Identifiable',
                    r'#\[derive\(Associations',
                    r'#\[derive\(QueryableByName',
                    r'#\[table_name',
                    r'#\[primary_key',
                    r'#\[belongs_to',
                    r'#\[column_name',
                    r'#\[diesel\(',
                    r'table!\s*\{',
                    r'joinable!\s*\(',
                    r'allow_tables_to_appear_in_same_query!\s*\(',
                    r'sql_function!\s*\{',
                    r'no_arg_sql_function!\s*\(',
                    r'schema::',
                    r'schema\.rs',
                    r'PgConnection',
                    r'MysqlConnection',
                    r'SqliteConnection',
                    r'Connection',
                    r'establish_connection',
                    r'connection\.execute',
                    r'connection\.transaction',
                    r'diesel::insert_into',
                    r'diesel::update',
                    r'diesel::delete',
                    r'diesel::select',
                    r'\.filter\(',
                    r'\.find\(',
                    r'\.order\(',
                    r'\.limit\(',
                    r'\.offset\(',
                    r'\.first\(',
                    r'\.get_result\(',
                    r'\.get_results\(',
                    r'\.load\(',
                    r'\.execute\(',
                    r'\.eq\(',
                    r'\.ne\(',
                    r'\.gt\(',
                    r'\.lt\(',
                    r'\.like\(',
                    r'\.not_like\(',
                    r'\.is_null\(',
                    r'\.is_not_null\(',
                    r'RunQueryDsl',
                    r'QueryDsl',
                    r'ExpressionMethods',
                    r'TextExpressionMethods',
                    r'BoolExpressionMethods',
                    r'migrations/',
                    r'diesel_migrations',
                    r'embed_migrations!',
                    r'run_pending_migrations',
                ],
                'files': ['schema.rs', 'models.rs', 'lib.rs'],
                'directories': ['src/', 'src/models/', 'migrations/'],
                'config_files': ['Cargo.toml', 'diesel.toml', '.env']
            },
            'serde': {
                'language': LanguageType.RUST,
                'patterns': [
                    r'serde::',
                    r'use\s+serde',
                    r'serde\s*=',
                    r'serde_json',
                    r'serde_yaml',
                    r'serde_toml',
                    r'serde_xml',
                    r'serde_derive',
                    r'#\[derive\(.*Serialize',
                    r'#\[derive\(.*Deserialize',
                    r'#\[serde\(',
                    r'#\[serde\(rename',
                    r'#\[serde\(rename_all',
                    r'#\[serde\(alias',
                    r'#\[serde\(default',
                    r'#\[serde\(skip',
                    r'#\[serde\(skip_serializing',
                    r'#\[serde\(skip_deserializing',
                    r'#\[serde\(skip_serializing_if',
                    r'#\[serde\(flatten',
                    r'#\[serde\(tag',
                    r'#\[serde\(content',
                    r'#\[serde\(untagged',
                    r'#\[serde\(bound',
                    r'#\[serde\(borrow',
                    r'#\[serde\(with',
                    r'#\[serde\(serialize_with',
                    r'#\[serde\(deserialize_with',
                    r'#\[serde\(from',
                    r'#\[serde\(into',
                    r'#\[serde\(try_from',
                    r'#\[serde\(remote',
                    r'Serialize',
                    r'Deserialize',
                    r'Serializer',
                    r'Deserializer',
                    r'ser::',
                    r'de::',
                    r'to_string\(',
                    r'to_string_pretty\(',
                    r'to_vec\(',
                    r'to_vec_pretty\(',
                    r'to_writer\(',
                    r'to_writer_pretty\(',
                    r'from_str\(',
                    r'from_slice\(',
                    r'from_reader\(',
                    r'from_value\(',
                    r'to_value\(',
                    r'json!\(',
                    r'Value',
                    r'Map<',
                    r'Number',
                ],
                'files': ['lib.rs', 'main.rs', 'models.rs'],
                'directories': ['src/', 'src/models/'],
                'config_files': ['Cargo.toml']
            },
            
            # C# Frameworks
            'aspnet-core': {
                'language': LanguageType.CSHARP,
                'patterns': [
                    r'using\s+Microsoft\.AspNetCore',
                    r'using\s+Microsoft\.Extensions',
                    r'WebApplication\.CreateBuilder',
                    r'WebApplication\.Create',
                    r'IApplicationBuilder',
                    r'IWebHostEnvironment',
                    r'IServiceCollection',
                    r'IConfiguration',
                    r'ConfigureServices\s*\(',
                    r'Configure\s*\(',
                    r'UseStartup<',
                    r'app\.Use',
                    r'app\.Map',
                    r'app\.Run',
                    r'app\.UseRouting',
                    r'app\.UseEndpoints',
                    r'app\.UseAuthentication',
                    r'app\.UseAuthorization',
                    r'app\.UseStaticFiles',
                    r'app\.UseCors',
                    r'app\.UseMiddleware',
                    r'app\.UseExceptionHandler',
                    r'app\.UseDeveloperExceptionPage',
                    r'app\.UseHttpsRedirection',
                    r'services\.AddControllers',
                    r'services\.AddMvc',
                    r'services\.AddRazorPages',
                    r'services\.AddSignalR',
                    r'services\.AddDbContext',
                    r'services\.AddScoped',
                    r'services\.AddTransient',
                    r'services\.AddSingleton',
                    r'services\.AddAuthentication',
                    r'services\.AddAuthorization',
                    r'services\.AddCors',
                    r'services\.AddHttpClient',
                    r'\[ApiController\]',
                    r'\[HttpGet\]',
                    r'\[HttpPost\]',
                    r'\[HttpPut\]',
                    r'\[HttpDelete\]',
                    r'\[HttpPatch\]',
                    r'\[Route\(',
                    r'\[Authorize\]',
                    r'\[AllowAnonymous\]',
                    r'\[FromBody\]',
                    r'\[FromQuery\]',
                    r'\[FromRoute\]',
                    r'\[FromForm\]',
                    r'\[FromServices\]',
                    r'ControllerBase',
                    r'Controller',
                    r'IActionResult',
                    r'ActionResult',
                    r'Ok\s*\(',
                    r'BadRequest\s*\(',
                    r'NotFound\s*\(',
                    r'Unauthorized\s*\(',
                    r'Forbid\s*\(',
                    r'CreatedAtAction\s*\(',
                    r'RedirectToAction\s*\(',
                    r'appsettings\.json',
                    r'launchSettings\.json',
                ],
                'files': [
                    'Program.cs',
                    'Startup.cs',
                    'appsettings.json',
                    'appsettings.Development.json',
                    'launchSettings.json'
                ],
                'directories': ['Controllers/', 'Models/', 'Views/', 'wwwroot/', 'Properties/'],
                'config_files': ['*.csproj', 'appsettings.json', 'web.config']
            },
            'entity-framework': {
                'language': LanguageType.CSHARP,
                'patterns': [
                    r'using\s+Microsoft\.EntityFrameworkCore',
                    r'using\s+System\.Data\.Entity',
                    r'DbContext',
                    r'DbSet<',
                    r'modelBuilder\.',
                    r'OnModelCreating\s*\(',
                    r'OnConfiguring\s*\(',
                    r'HasMany\s*\(',
                    r'HasOne\s*\(',
                    r'WithMany\s*\(',
                    r'WithOne\s*\(',
                    r'HasKey\s*\(',
                    r'HasIndex\s*\(',
                    r'HasForeignKey\s*\(',
                    r'HasAlternateKey\s*\(',
                    r'Property\s*\(',
                    r'IsRequired\s*\(',
                    r'HasMaxLength\s*\(',
                    r'HasColumnName\s*\(',
                    r'HasColumnType\s*\(',
                    r'HasDefaultValue\s*\(',
                    r'HasDefaultValueSql\s*\(',
                    r'ValueGeneratedOnAdd\s*\(',
                    r'ValueGeneratedOnUpdate\s*\(',
                    r'IsConcurrencyToken\s*\(',
                    r'UseIdentityColumn\s*\(',
                    r'ToTable\s*\(',
                    r'HasDiscriminator\s*\(',
                    r'HasQueryFilter\s*\(',
                    r'HasData\s*\(',
                    r'Migration',
                    r'migrationBuilder\.',
                    r'CreateTable\s*\(',
                    r'DropTable\s*\(',
                    r'AddColumn\s*\(',
                    r'DropColumn\s*\(',
                    r'CreateIndex\s*\(',
                    r'DropIndex\s*\(',
                    r'\.Include\s*\(',
                    r'\.ThenInclude\s*\(',
                    r'\.Where\s*\(',
                    r'\.Select\s*\(',
                    r'\.OrderBy\s*\(',
                    r'\.OrderByDescending\s*\(',
                    r'\.FirstOrDefault\s*\(',
                    r'\.SingleOrDefault\s*\(',
                    r'\.ToList\s*\(',
                    r'\.ToListAsync\s*\(',
                    r'\.AsNoTracking\s*\(',
                    r'\.AsQueryable\s*\(',
                    r'SaveChanges\s*\(',
                    r'SaveChangesAsync\s*\(',
                    r'Add\s*\(',
                    r'Update\s*\(',
                    r'Remove\s*\(',
                    r'Attach\s*\(',
                ],
                'files': ['*Context.cs', 'Migrations/*.cs'],
                'directories': ['Migrations/', 'Models/', 'Data/'],
                'config_files': ['*.csproj', 'appsettings.json']
            },
            'wpf': {
                'language': LanguageType.CSHARP,
                'patterns': [
                    r'using\s+System\.Windows',
                    r'<Window',
                    r'<UserControl',
                    r'<Application',
                    r'<Page',
                    r'<ResourceDictionary',
                    r'xmlns="http://schemas\.microsoft\.com/winfx/2006/xaml/presentation"',
                    r'xmlns:x="http://schemas\.microsoft\.com/winfx/2006/xaml"',
                    r'x:Class=',
                    r'x:Name=',
                    r'Window',
                    r'UserControl',
                    r'Application',
                    r'Page',
                    r'FrameworkElement',
                    r'UIElement',
                    r'DependencyObject',
                    r'DependencyProperty',
                    r'INotifyPropertyChanged',
                    r'PropertyChanged',
                    r'ICommand',
                    r'RelayCommand',
                    r'DelegateCommand',
                    r'ObservableCollection',
                    r'Binding',
                    r'DataContext',
                    r'<Grid>',
                    r'<StackPanel>',
                    r'<DockPanel>',
                    r'<Canvas>',
                    r'<WrapPanel>',
                    r'<Button',
                    r'<TextBox',
                    r'<TextBlock',
                    r'<Label',
                    r'<ComboBox',
                    r'<ListBox',
                    r'<ListView',
                    r'<DataGrid',
                    r'<TreeView',
                    r'<Menu',
                    r'<ToolBar',
                    r'<StatusBar',
                    r'Click=',
                    r'Command=',
                    r'CommandParameter=',
                    r'ItemsSource=',
                    r'SelectedItem=',
                    r'Text=',
                    r'Content=',
                    r'Style=',
                    r'Template=',
                    r'Trigger',
                    r'Converter',
                    r'RoutedEventArgs',
                    r'MessageBox\.Show',
                    r'\.xaml',
                    r'\.xaml\.cs',
                ],
                'files': [
                    'App.xaml',
                    'App.xaml.cs',
                    'MainWindow.xaml',
                    'MainWindow.xaml.cs',
                    '*.xaml',
                    '*.xaml.cs'
                ],
                'directories': ['Views/', 'ViewModels/', 'Models/', 'Controls/', 'Resources/'],
                'config_files': ['*.csproj', 'App.config', 'App.xaml']
            },
            'xamarin': {
                'language': LanguageType.CSHARP,
                'patterns': [
                    r'using\s+Xamarin\.Forms',
                    r'using\s+Xamarin\.Essentials',
                    r'Xamarin\.Forms\.Application',
                    r'ContentPage',
                    r'ContentView',
                    r'NavigationPage',
                    r'TabbedPage',
                    r'CarouselPage',
                    r'MasterDetailPage',
                    r'FlyoutPage',
                    r'Shell',
                    r'BindableObject',
                    r'BindableProperty',
                    r'BindingContext',
                    r'INotifyPropertyChanged',
                    r'ObservableCollection',
                    r'Command',
                    r'ICommand',
                    r'<ContentPage',
                    r'<ContentView',
                    r'<StackLayout>',
                    r'<Grid>',
                    r'<ScrollView>',
                    r'<Frame>',
                    r'<AbsoluteLayout>',
                    r'<RelativeLayout>',
                    r'<FlexLayout>',
                    r'<Label',
                    r'<Button',
                    r'<Entry',
                    r'<Editor',
                    r'<Picker',
                    r'<DatePicker',
                    r'<TimePicker',
                    r'<Switch',
                    r'<Slider',
                    r'<Stepper',
                    r'<ListView',
                    r'<CollectionView',
                    r'<CarouselView',
                    r'<TableView',
                    r'<Image',
                    r'<BoxView',
                    r'<WebView',
                    r'<Map',
                    r'Text=',
                    r'Command=',
                    r'ItemsSource=',
                    r'SelectedItem=',
                    r'Clicked=',
                    r'Tapped=',
                    r'Navigation\.PushAsync',
                    r'Navigation\.PopAsync',
                    r'Navigation\.PushModalAsync',
                    r'Navigation\.PopModalAsync',
                    r'DisplayAlert',
                    r'DisplayActionSheet',
                    r'DependencyService',
                    r'MessagingCenter',
                    r'Device\.BeginInvokeOnMainThread',
                    r'Device\.RuntimePlatform',
                    r'OnPlatform',
                    r'Xamarin\.iOS',
                    r'Xamarin\.Android',
                    r'Xamarin\.Mac',
                ],
                'files': [
                    'App.xaml',
                    'App.xaml.cs',
                    'MainPage.xaml',
                    'MainPage.xaml.cs',
                    'AssemblyInfo.cs'
                ],
                'directories': [
                    'Views/',
                    'ViewModels/',
                    'Models/',
                    'Services/',
                    'Controls/',
                    'Platforms/Android/',
                    'Platforms/iOS/'
                ],
                'config_files': ['*.csproj', 'App.xaml']
            },
            
            # Ruby Frameworks
            'rails': {
                'language': LanguageType.RUBY,
                'patterns': [
                    r'Rails\.',
                    r'::Rails',
                    r'ActionController',
                    r'ActiveRecord',
                    r'ActionView',
                    r'ActiveSupport',
                    r'ActionMailer',
                    r'ActiveJob',
                    r'ActionCable',
                    r'ActiveStorage',
                    r'ApplicationController',
                    r'ApplicationRecord',
                    r'ApplicationMailer',
                    r'ApplicationJob',
                    r'ApplicationCable',
                    r'Rails\.application',
                    r'Rails\.root',
                    r'Rails\.env',
                    r'Rails\.logger',
                    r'config\.routes',
                    r'resources\s+:',
                    r'resource\s+:',
                    r'get\s+["\']/',
                    r'post\s+["\']/',
                    r'put\s+["\']/',
                    r'patch\s+["\']/',
                    r'delete\s+["\']/',
                    r'root\s+to:',
                    r'root\s+["\']',
                    r'before_action',
                    r'after_action',
                    r'around_action',
                    r'skip_before_action',
                    r'before_filter',
                    r'after_filter',
                    r'respond_to',
                    r'render\s+:',
                    r'render\s+json:',
                    r'render\s+xml:',
                    r'redirect_to',
                    r'params\[:',
                    r'params\.require',
                    r'params\.permit',
                    r'has_many',
                    r'belongs_to',
                    r'has_one',
                    r'has_and_belongs_to_many',
                    r'validates',
                    r'validates_presence_of',
                    r'validates_uniqueness_of',
                    r'validates_length_of',
                    r'scope\s+:',
                    r'default_scope',
                    r'where\(',
                    r'order\(',
                    r'limit\(',
                    r'includes\(',
                    r'joins\(',
                    r'group\(',
                    r'having\(',
                    r'find_by',
                    r'find_or_create_by',
                    r'create!',
                    r'update!',
                    r'destroy!',
                    r'save!',
                    r't\.string',
                    r't\.integer',
                    r't\.text',
                    r't\.datetime',
                    r't\.boolean',
                    r't\.references',
                    r't\.index',
                    r'add_column',
                    r'remove_column',
                    r'add_index',
                    r'remove_index',
                    r'create_table',
                    r'drop_table',
                    r'migration',
                    r'schema\.rb',
                    r'Gemfile',
                    r'gem\s+["\']rails["\']',
                ],
                'files': [
                    'Gemfile',
                    'Gemfile.lock',
                    'Rakefile',
                    'config.ru',
                    'config/routes.rb',
                    'config/application.rb',
                    'config/environment.rb',
                    'config/database.yml',
                    'db/schema.rb',
                    'db/seeds.rb'
                ],
                'directories': [
                    'app/',
                    'app/controllers/',
                    'app/models/',
                    'app/views/',
                    'app/helpers/',
                    'app/assets/',
                    'app/mailers/',
                    'app/jobs/',
                    'app/channels/',
                    'config/',
                    'db/',
                    'db/migrate/',
                    'public/',
                    'vendor/',
                    'test/',
                    'spec/'
                ],
                'config_files': ['Gemfile', 'config/database.yml', 'config/routes.rb']
            },
            'sinatra': {
                'language': LanguageType.RUBY,
                'patterns': [
                    r'require\s+["\']sinatra["\']',
                    r'require_relative.*sinatra',
                    r'Sinatra::Base',
                    r'Sinatra::Application',
                    r'get\s+["\']/',
                    r'post\s+["\']/',
                    r'put\s+["\']/',
                    r'patch\s+["\']/',
                    r'delete\s+["\']/',
                    r'options\s+["\']/',
                    r'head\s+["\']/',
                    r'link\s+["\']/',
                    r'unlink\s+["\']/',
                    r'before\s+do',
                    r'after\s+do',
                    r'helpers\s+do',
                    r'configure\s+do',
                    r'configure\s+:',
                    r'set\s+:',
                    r'enable\s+:',
                    r'disable\s+:',
                    r'use\s+Rack::',
                    r'run\s+Sinatra::Application',
                    r'params\[:',
                    r'request\.',
                    r'response\.',
                    r'session\[:',
                    r'halt\s+',
                    r'pass',
                    r'redirect\s+',
                    r'send_file',
                    r'attachment',
                    r'erb\s+:',
                    r'haml\s+:',
                    r'slim\s+:',
                    r'json\s+',
                    r'content_type',
                    r'status\s+',
                    r'headers\[',
                    r'error\s+\d+\s+do',
                    r'not_found\s+do',
                    r'Sinatra::Reloader',
                    r'register\s+Sinatra::',
                    r'gem\s+["\']sinatra["\']',
                ],
                'files': [
                    'app.rb',
                    'application.rb',
                    'config.ru',
                    'Gemfile',
                    'Rakefile'
                ],
                'directories': ['views/', 'public/', 'lib/', 'config/'],
                'config_files': ['Gemfile', 'config.ru']
            },
            'hanami': {
                'language': LanguageType.RUBY,
                'patterns': [
                    r'Hanami\.',
                    r'Hanami::',
                    r'require\s+["\']hanami["\']',
                    r'Hanami::Application',
                    r'Hanami::Action',
                    r'Hanami::Controller',
                    r'Hanami::View',
                    r'Hanami::Repository',
                    r'Hanami::Entity',
                    r'Hanami::Interactor',
                    r'Hanami::Mailer',
                    r'Hanami::Router',
                    r'Hanami::Helpers',
                    r'Hanami::Assets',
                    r'expose\s+:',
                    r'include\s+Hanami::Action',
                    r'include\s+Hanami::View',
                    r'self\.call',
                    r'routes\s+do',
                    r'get\s+["\']/',
                    r'post\s+["\']/',
                    r'put\s+["\']/',
                    r'patch\s+["\']/',
                    r'delete\s+["\']/',
                    r'resource\s+:',
                    r'resources\s+:',
                    r'mount\s+',
                    r'root\s+to:',
                    r'params\[:',
                    r'validate\s+:',
                    r'required\(',
                    r'optional\(',
                    r'render\s+view:',
                    r'redirect_to',
                    r'halt\s+',
                    r'handle_exception',
                    r'gem\s+["\']hanami["\']',
                ],
                'files': [
                    'config.ru',
                    'Gemfile',
                    'Rakefile',
                    'config/environment.rb',
                    'config/routes.rb'
                ],
                'directories': [
                    'apps/',
                    'lib/',
                    'db/',
                    'config/',
                    'spec/',
                    'public/'
                ],
                'config_files': ['Gemfile', 'config/environment.rb', '.hanamirc']
            },
            'grape': {
                'language': LanguageType.RUBY,
                'patterns': [
                    r'Grape::API',
                    r'require\s+["\']grape["\']',
                    r'class.*<\s*Grape::API',
                    r'format\s+:json',
                    r'format\s+:xml',
                    r'prefix\s+["\']api["\']',
                    r'version\s+["\']v\d+["\']',
                    r'namespace\s+:',
                    r'resource\s+:',
                    r'desc\s+["\']',
                    r'params\s+do',
                    r'requires\s+:',
                    r'optional\s+:',
                    r'group\s+:',
                    r'exactly_one_of',
                    r'at_least_one_of',
                    r'mutually_exclusive',
                    r'get\s+do',
                    r'post\s+do',
                    r'put\s+do',
                    r'patch\s+do',
                    r'delete\s+do',
                    r'head\s+do',
                    r'options\s+do',
                    r'route_param\s+:',
                    r'helpers\s+do',
                    r'before\s+do',
                    r'after\s+do',
                    r'rescue_from',
                    r'error!',
                    r'present\s+',
                    r'entity\s+',
                    r'expose\s+:',
                    r'authenticate!',
                    r'current_user',
                    r'declared\(',
                    r'permitted_params',
                    r'content_type\s+:',
                    r'status\s+\d+',
                    r'header\s+["\']',
                    r'route\.route_',
                    r'gem\s+["\']grape["\']',
                ],
                'files': [
                    'config.ru',
                    'Gemfile',
                    'Rakefile',
                    'api.rb'
                ],
                'directories': [
                    'api/',
                    'app/api/',
                    'lib/api/',
                    'spec/'
                ],
                'config_files': ['Gemfile', 'config.ru']
            },
            'roda': {
                'language': LanguageType.RUBY,
                'patterns': [
                    r'Roda',
                    r'require\s+["\']roda["\']',
                    r'class.*<\s*Roda',
                    r'plugin\s+:',
                    r'route\s+do\s*\|r\|',
                    r'r\.root',
                    r'r\.get',
                    r'r\.post',
                    r'r\.put',
                    r'r\.patch',
                    r'r\.delete',
                    r'r\.is',
                    r'r\.on',
                    r'r\.redirect',
                    r'r\.halt',
                    r'r\.params',
                    r'r\.session',
                    r'r\.request',
                    r'r\.response',
                    r'r\.scope',
                    r'r\.multi_route',
                    r'response\[:',
                    r'request\.',
                    r'view\s+["\']',
                    r'render\s+["\']',
                    r'partial\s+["\']',
                    r'layout\s+["\']',
                    r'opts\[:',
                    r'freeze_app',
                    r'compile_routes!',
                    r'use\s+Rack::',
                    r'gem\s+["\']roda["\']',
                ],
                'files': [
                    'config.ru',
                    'app.rb',
                    'Gemfile',
                    'Rakefile'
                ],
                'directories': [
                    'routes/',
                    'views/',
                    'public/',
                    'models/',
                    'lib/'
                ],
                'config_files': ['Gemfile', 'config.ru']
            },
            
            # PHP Frameworks
            'laravel': {
                'language': LanguageType.PHP,
                'patterns': [
                    r'namespace\s+App\\\\',
                    r'use\s+Illuminate\\\\',
                    r'use\s+Laravel\\\\',
                    r'class.*extends.*Controller',
                    r'class.*extends.*Model',
                    r'class.*extends.*Seeder',
                    r'class.*extends.*Migration',
                    r'class.*extends.*Request',
                    r'class.*extends.*Middleware',
                    r'Route::',
                    r'Route::get\(',
                    r'Route::post\(',
                    r'Route::put\(',
                    r'Route::patch\(',
                    r'Route::delete\(',
                    r'Route::resource\(',
                    r'Route::apiResource\(',
                    r'Route::group\(',
                    r'Route::middleware\(',
                    r'Route::prefix\(',
                    r'Route::name\(',
                    r'Auth::',
                    r'Auth::user\(',
                    r'Auth::check\(',
                    r'Auth::login\(',
                    r'Auth::logout\(',
                    r'DB::',
                    r'DB::table\(',
                    r'DB::select\(',
                    r'DB::transaction\(',
                    r'Schema::',
                    r'Schema::create\(',
                    r'Schema::table\(',
                    r'Schema::drop\(',
                    r'\$table->',
                    r'\$table->id\(',
                    r'\$table->string\(',
                    r'\$table->integer\(',
                    r'\$table->boolean\(',
                    r'\$table->timestamps\(',
                    r'\$table->foreign\(',
                    r'\$table->index\(',
                    r'Eloquent',
                    r'\$this->hasMany\(',
                    r'\$this->belongsTo\(',
                    r'\$this->belongsToMany\(',
                    r'\$this->hasOne\(',
                    r'\$this->morphMany\(',
                    r'\$this->morphTo\(',
                    r'\$fillable',
                    r'\$guarded',
                    r'\$hidden',
                    r'\$casts',
                    r'request\(\)',
                    r'response\(\)',
                    r'redirect\(\)',
                    r'view\(',
                    r'compact\(',
                    r'with\(',
                    r'validate\(',
                    r'validated\(',
                    r'Validator::',
                    r'Storage::',
                    r'File::',
                    r'Cache::',
                    r'Session::',
                    r'Cookie::',
                    r'Mail::',
                    r'Queue::',
                    r'Event::',
                    r'Log::',
                    r'Artisan::',
                    r'Config::',
                    r'App::',
                    r'Blade',
                    r'@extends',
                    r'@section',
                    r'@yield',
                    r'@include',
                    r'@if',
                    r'@foreach',
                    r'@forelse',
                    r'@while',
                    r'@switch',
                    r'@auth',
                    r'@guest',
                    r'@can',
                    r'@cannot',
                    r'@csrf',
                    r'@method',
                    r'{{.*}}',
                    r'{!!.*!!}',
                    r'composer\.json',
                    r'artisan',
                ],
                'files': [
                    'composer.json',
                    'composer.lock',
                    'artisan',
                    '.env',
                    '.env.example',
                    'webpack.mix.js',
                    'package.json'
                ],
                'directories': [
                    'app/',
                    'app/Http/',
                    'app/Http/Controllers/',
                    'app/Models/',
                    'app/Http/Middleware/',
                    'app/Http/Requests/',
                    'app/Providers/',
                    'bootstrap/',
                    'config/',
                    'database/',
                    'database/migrations/',
                    'database/seeders/',
                    'public/',
                    'resources/',
                    'resources/views/',
                    'routes/',
                    'storage/',
                    'tests/'
                ],
                'config_files': ['composer.json', '.env', 'config/app.php']
            },
            'symfony': {
                'language': LanguageType.PHP,
                'patterns': [
                    r'namespace\s+App\\\\',
                    r'namespace\s+Symfony\\\\',
                    r'use\s+Symfony\\\\',
                    r'use\s+Doctrine\\\\',
                    r'use\s+Sensio\\\\',
                    r'extends\s+AbstractController',
                    r'extends\s+Controller',
                    r'#\[Route\(',
                    r'@Route\(',
                    r'#\[Entity',
                    r'@Entity',
                    r'#\[Column\(',
                    r'@Column\(',
                    r'#\[Id\]',
                    r'@Id',
                    r'#\[GeneratedValue',
                    r'@GeneratedValue',
                    r'#\[OneToMany',
                    r'@OneToMany',
                    r'#\[ManyToOne',
                    r'@ManyToOne',
                    r'#\[ManyToMany',
                    r'@ManyToMany',
                    r'#\[OneToOne',
                    r'@OneToOne',
                    r'#\[JoinColumn',
                    r'@JoinColumn',
                    r'#\[Table\(',
                    r'@Table\(',
                    r'#\[Index\(',
                    r'@Index\(',
                    r'#\[UniqueConstraint',
                    r'@UniqueConstraint',
                    r'\$this->render\(',
                    r'\$this->renderView\(',
                    r'\$this->json\(',
                    r'\$this->redirect\(',
                    r'\$this->redirectToRoute\(',
                    r'\$this->forward\(',
                    r'\$this->createForm\(',
                    r'\$this->createFormBuilder\(',
                    r'\$this->getDoctrine\(',
                    r'\$this->getParameter\(',
                    r'\$this->get\(',
                    r'\$this->has\(',
                    r'\$this->addFlash\(',
                    r'\$this->isGranted\(',
                    r'\$this->denyAccessUnlessGranted\(',
                    r'\$this->createAccessDeniedException\(',
                    r'\$this->createNotFoundException\(',
                    r'Request\s+\$request',
                    r'Response',
                    r'JsonResponse',
                    r'RedirectResponse',
                    r'EntityManagerInterface',
                    r'ManagerRegistry',
                    r'FormFactoryInterface',
                    r'FormInterface',
                    r'ValidatorInterface',
                    r'TranslatorInterface',
                    r'LoggerInterface',
                    r'EventDispatcherInterface',
                    r'ContainerInterface',
                    r'KernelInterface',
                    r'services\.yaml',
                    r'services\.yml',
                    r'config\.yaml',
                    r'config\.yml',
                    r'routing\.yaml',
                    r'routing\.yml',
                    r'security\.yaml',
                    r'security\.yml',
                    r'doctrine\.yaml',
                    r'doctrine\.yml',
                    r'twig\.yaml',
                    r'twig\.yml',
                    r'framework\.yaml',
                    r'framework\.yml',
                    r'composer\.json',
                    r'symfony/.*',
                    r'bin/console',
                ],
                'files': [
                    'composer.json',
                    'composer.lock',
                    'symfony.lock',
                    '.env',
                    '.env.local',
                    'bin/console',
                    'public/index.php'
                ],
                'directories': [
                    'src/',
                    'src/Controller/',
                    'src/Entity/',
                    'src/Repository/',
                    'src/Form/',
                    'src/Service/',
                    'src/EventListener/',
                    'src/EventSubscriber/',
                    'src/Command/',
                    'src/Security/',
                    'src/Twig/',
                    'config/',
                    'config/packages/',
                    'config/routes/',
                    'public/',
                    'templates/',
                    'translations/',
                    'var/',
                    'vendor/',
                    'tests/'
                ],
                'config_files': ['composer.json', '.env', 'config/services.yaml']
            },
            'codeigniter': {
                'language': LanguageType.PHP,
                'patterns': [
                    r'CI_Controller',
                    r'CI_Model',
                    r'CodeIgniter',
                    r'extends\s+CI_Controller',
                    r'extends\s+CI_Model',
                    r'extends\s+Controller',
                    r'extends\s+Model',
                    r'\$this->load->',
                    r'\$this->load->model\(',
                    r'\$this->load->view\(',
                    r'\$this->load->library\(',
                    r'\$this->load->helper\(',
                    r'\$this->load->database\(',
                    r'\$this->db->',
                    r'\$this->db->get\(',
                    r'\$this->db->select\(',
                    r'\$this->db->from\(',
                    r'\$this->db->where\(',
                    r'\$this->db->join\(',
                    r'\$this->db->insert\(',
                    r'\$this->db->update\(',
                    r'\$this->db->delete\(',
                    r'\$this->db->query\(',
                    r'\$this->input->',
                    r'\$this->input->post\(',
                    r'\$this->input->get\(',
                    r'\$this->input->cookie\(',
                    r'\$this->input->server\(',
                    r'\$this->output->',
                    r'\$this->output->set_content_type\(',
                    r'\$this->output->set_output\(',
                    r'\$this->session->',
                    r'\$this->session->userdata\(',
                    r'\$this->session->set_userdata\(',
                    r'\$this->session->flashdata\(',
                    r'\$this->session->set_flashdata\(',
                    r'\$this->form_validation->',
                    r'\$this->form_validation->set_rules\(',
                    r'\$this->form_validation->run\(',
                    r'\$this->upload->',
                    r'\$this->email->',
                    r'\$this->pagination->',
                    r'\$this->security->',
                    r'\$this->uri->',
                    r'\$this->uri->segment\(',
                    r'\$this->config->',
                    r'\$this->config->item\(',
                    r'site_url\(',
                    r'base_url\(',
                    r'redirect\(',
                    r'anchor\(',
                    r'form_open\(',
                    r'form_close\(',
                    r'form_input\(',
                    r'form_dropdown\(',
                    r'form_submit\(',
                    r'set_value\(',
                    r'form_error\(',
                    r'validation_errors\(',
                    r'show_error\(',
                    r'show_404\(',
                    r'log_message\(',
                    r'BASEPATH',
                    r'APPPATH',
                    r'FCPATH',
                    r'index\.php',
                    r'system/',
                    r'application/',
                ],
                'files': [
                    'index.php',
                    'composer.json',
                    '.htaccess'
                ],
                'directories': [
                    'application/',
                    'application/controllers/',
                    'application/models/',
                    'application/views/',
                    'application/libraries/',
                    'application/helpers/',
                    'application/config/',
                    'application/hooks/',
                    'application/language/',
                    'application/third_party/',
                    'system/',
                    'public/',
                    'writable/'
                ],
                'config_files': ['application/config/config.php', 'application/config/database.php']
            },
            'slim': {
                'language': LanguageType.PHP,
                'patterns': [
                    r'Slim\\\\App',
                    r'Slim\\\\Factory\\\\AppFactory',
                    r'use\s+Slim\\\\',
                    r'use\s+Psr\\\\',
                    r'AppFactory::create\(',
                    r'new\s+\\\\Slim\\\\App',
                    r'\$app->get\(',
                    r'\$app->post\(',
                    r'\$app->put\(',
                    r'\$app->patch\(',
                    r'\$app->delete\(',
                    r'\$app->options\(',
                    r'\$app->any\(',
                    r'\$app->map\(',
                    r'\$app->group\(',
                    r'\$app->redirect\(',
                    r'\$app->run\(',
                    r'\$app->add\(',
                    r'\$app->addMiddleware\(',
                    r'\$app->addRoutingMiddleware\(',
                    r'\$app->addErrorMiddleware\(',
                    r'\$app->addBodyParsingMiddleware\(',
                    r'Request\s+\$request',
                    r'Response\s+\$response',
                    r'ServerRequestInterface',
                    r'ResponseInterface',
                    r'\$request->getMethod\(',
                    r'\$request->getUri\(',
                    r'\$request->getQueryParams\(',
                    r'\$request->getParsedBody\(',
                    r'\$request->getUploadedFiles\(',
                    r'\$request->getAttribute\(',
                    r'\$response->getBody\(\)->write\(',
                    r'\$response->withStatus\(',
                    r'\$response->withHeader\(',
                    r'\$response->withJson\(',
                    r'withAttribute\(',
                    r'getAttribute\(',
                    r'getAttributes\(',
                    r'Container',
                    r'ContainerInterface',
                    r'DI\\\\Container',
                    r'Middleware',
                    r'MiddlewareInterface',
                    r'ErrorHandler',
                    r'ErrorMiddleware',
                    r'composer\.json',
                    r'slim/slim',
                ],
                'files': [
                    'composer.json',
                    'composer.lock',
                    'public/index.php',
                    'index.php'
                ],
                'directories': [
                    'src/',
                    'app/',
                    'public/',
                    'config/',
                    'routes/',
                    'middleware/',
                    'templates/'
                ],
                'config_files': ['composer.json', 'config/settings.php']
            },
            'yii': {
                'language': LanguageType.PHP,
                'patterns': [
                    r'yii\\\\',
                    r'use\s+yii\\\\',
                    r'Yii::',
                    r'Yii::\$app',
                    r'extends\s+Controller',
                    r'extends\s+ActiveRecord',
                    r'extends\s+Model',
                    r'extends\s+Widget',
                    r'extends\s+Component',
                    r'extends\s+Module',
                    r'extends\s+Migration',
                    r'yii\\\\web\\\\Controller',
                    r'yii\\\\db\\\\ActiveRecord',
                    r'yii\\\\base\\\\Model',
                    r'yii\\\\base\\\\Widget',
                    r'yii\\\\base\\\\Component',
                    r'yii\\\\base\\\\Module',
                    r'yii\\\\db\\\\Migration',
                    r'public\s+function\s+actionIndex',
                    r'public\s+function\s+action',
                    r'public\s+function\s+behaviors\(',
                    r'public\s+function\s+rules\(',
                    r'public\s+function\s+attributeLabels\(',
                    r'public\s+function\s+scenarios\(',
                    r'\$this->render\(',
                    r'\$this->renderPartial\(',
                    r'\$this->renderAjax\(',
                    r'\$this->redirect\(',
                    r'\$this->refresh\(',
                    r'\$this->goHome\(',
                    r'\$this->goBack\(',
                    r'Html::',
                    r'Html::encode\(',
                    r'Html::a\(',
                    r'Html::beginForm\(',
                    r'Html::endForm\(',
                    r'Html::input\(',
                    r'Html::submitButton\(',
                    r'ActiveForm::begin\(',
                    r'ActiveForm::end\(',
                    r'\$form->field\(',
                    r'GridView::widget\(',
                    r'ListView::widget\(',
                    r'DetailView::widget\(',
                    r'find\(\)',
                    r'findOne\(',
                    r'findAll\(',
                    r'where\(',
                    r'andWhere\(',
                    r'orWhere\(',
                    r'orderBy\(',
                    r'limit\(',
                    r'offset\(',
                    r'one\(\)',
                    r'all\(\)',
                    r'count\(\)',
                    r'save\(\)',
                    r'delete\(\)',
                    r'hasOne\(',
                    r'hasMany\(',
                    r'load\(',
                    r'validate\(',
                    r'beforeSave\(',
                    r'afterSave\(',
                    r'beforeDelete\(',
                    r'afterDelete\(',
                    r'tableName\(',
                    r'getDb\(',
                    r'createCommand\(',
                    r'transaction\(',
                    r'composer\.json',
                    r'yiisoft/yii2',
                ],
                'files': [
                    'composer.json',
                    'composer.lock',
                    'yii',
                    'web/index.php',
                    'requirements.php'
                ],
                'directories': [
                    'config/',
                    'controllers/',
                    'models/',
                    'views/',
                    'web/',
                    'runtime/',
                    'vendor/',
                    'migrations/',
                    'commands/',
                    'components/',
                    'widgets/',
                    'assets/',
                    'tests/'
                ],
                'config_files': ['composer.json', 'config/web.php', 'config/db.php']
            },
            
            # Swift Frameworks
            'vapor': {
                'language': LanguageType.SWIFT,
                'patterns': [
                    r'import\s+Vapor',
                    r'import\s+Fluent',
                    r'import\s+Leaf',
                    r'Application',
                    r'Request',
                    r'Response',
                    r'Routes',
                    r'app\.get\(',
                    r'app\.post\(',
                    r'app\.put\(',
                    r'app\.patch\(',
                    r'app\.delete\(',
                    r'app\.on\(',
                    r'app\.grouped\(',
                    r'app\.middleware\.use\(',
                    r'routes\(',
                    r'req\.parameters\.get\(',
                    r'req\.query\[',
                    r'req\.content\.decode\(',
                    r'req\.body\.collect\(',
                    r'req\.headers',
                    r'req\.cookies',
                    r'req\.session',
                    r'Response\(status:',
                    r'\.encode\(status:',
                    r'\.redirect\(to:',
                    r'EventLoopFuture',
                    r'EventLoop',
                    r'Promise',
                    r'\.flatMap',
                    r'\.map',
                    r'\.transform',
                    r'\.unwrap',
                    r'\.wait\(',
                    r'Model',
                    r'Content',
                    r'Migration',
                    r'@ID',
                    r'@Field',
                    r'@Parent',
                    r'@Children',
                    r'@Siblings',
                    r'@Timestamp',
                    r'@Enum',
                    r'@OptionalField',
                    r'@OptionalParent',
                    r'@OptionalChild',
                    r'schema\(',
                    r'database\(',
                    r'\.field\(',
                    r'\.unique\(',
                    r'\.references\(',
                    r'\.create\(',
                    r'\.update\(',
                    r'\.delete\(',
                    r'\.query\(',
                    r'\.filter\(',
                    r'\.sort\(',
                    r'\.range\(',
                    r'\.first\(',
                    r'\.all\(',
                    r'\.count\(',
                    r'\.save\(',
                    r'\.delete\(',
                    r'\.create\(on:',
                    r'\.update\(on:',
                    r'Middleware',
                    r'Authenticatable',
                    r'SessionAuthenticatable',
                    r'BearerAuthenticatable',
                    r'BasicAuthenticatable',
                    r'JWTPayload',
                    r'configure\.swift',
                    r'routes\.swift',
                    r'Package\.swift',
                ],
                'files': [
                    'Package.swift',
                    'Sources/Run/main.swift',
                    'Sources/App/configure.swift',
                    'Sources/App/routes.swift',
                    'Sources/App/boot.swift'
                ],
                'directories': [
                    'Sources/',
                    'Sources/App/',
                    'Sources/App/Controllers/',
                    'Sources/App/Models/',
                    'Sources/App/Migrations/',
                    'Sources/App/Middleware/',
                    'Sources/Run/',
                    'Tests/',
                    'Resources/',
                    'Resources/Views/',
                    'Public/'
                ],
                'config_files': ['Package.swift', '.env', '.env.development']
            },
            'perfect': {
                'language': LanguageType.SWIFT,
                'patterns': [
                    r'import\s+PerfectHTTP',
                    r'import\s+PerfectHTTPServer',
                    r'import\s+PerfectLib',
                    r'import\s+PerfectNet',
                    r'import\s+PerfectThread',
                    r'import\s+PerfectCrypto',
                    r'import\s+PerfectSession',
                    r'HTTPServer',
                    r'HTTPRequest',
                    r'HTTPResponse',
                    r'Routes',
                    r'Route',
                    r'RouteHandler',
                    r'routes\.add\(',
                    r'method:\s*\.',
                    r'uri:',
                    r'handler:',
                    r'request\.method',
                    r'request\.path',
                    r'request\.queryParams',
                    r'request\.postParams',
                    r'request\.param\(',
                    r'request\.params\(',
                    r'request\.header\(',
                    r'request\.headers',
                    r'response\.status',
                    r'response\.setHeader\(',
                    r'response\.setBody\(',
                    r'response\.appendBody\(',
                    r'response\.completed\(',
                    r'HTTPResponseStatus',
                    r'HTTPMethod',
                    r'server\.serverPort',
                    r'server\.serverAddress',
                    r'server\.serverName',
                    r'server\.documentRoot',
                    r'server\.addRoutes\(',
                    r'server\.start\(',
                    r'makeRoutes\(',
                    r'baseRoutes\(',
                    r'StaticFileHandler',
                    r'SessionManager',
                    r'Package\.swift',
                    r'PerfectHTTPServer',
                ],
                'files': [
                    'Package.swift',
                    'Sources/main.swift',
                    'Sources/PerfectApp/main.swift'
                ],
                'directories': [
                    'Sources/',
                    'Sources/PerfectApp/',
                    'Tests/',
                    'webroot/',
                    'Config/'
                ],
                'config_files': ['Package.swift']
            },
            'kitura': {
                'language': LanguageType.SWIFT,
                'patterns': [
                    r'import\s+Kitura',
                    r'import\s+KituraNet',
                    r'import\s+KituraSession',
                    r'import\s+KituraStencil',
                    r'import\s+SwiftyJSON',
                    r'Router',
                    r'RouterRequest',
                    r'RouterResponse',
                    r'RouterMiddleware',
                    r'router\.get\(',
                    r'router\.post\(',
                    r'router\.put\(',
                    r'router\.patch\(',
                    r'router\.delete\(',
                    r'router\.all\(',
                    r'router\.route\(',
                    r'router\.use\(',
                    r'request\.parameters',
                    r'request\.queryParameters',
                    r'request\.body',
                    r'request\.headers',
                    r'request\.cookies',
                    r'request\.session',
                    r'response\.send\(',
                    r'response\.status\(',
                    r'response\.headers',
                    r'response\.redirect\(',
                    r'response\.render\(',
                    r'response\.json\(',
                    r'next\(\)',
                    r'BodyParser',
                    r'StaticFileServer',
                    r'Session',
                    r'Credentials',
                    r'CodableRouter',
                    r'Codable',
                    r'TypeSafeMiddleware',
                    r'RequestError',
                    r'Kitura\.addHTTPServer\(',
                    r'Kitura\.run\(',
                    r'Kitura\.start\(',
                    r'Kitura\.stop\(',
                    r'Package\.swift',
                    r'IBM-Swift/Kitura',
                ],
                'files': [
                    'Package.swift',
                    'Sources/Application/Application.swift',
                    'Sources/Application/Routes.swift'
                ],
                'directories': [
                    'Sources/',
                    'Sources/Application/',
                    'Sources/Application/Routes/',
                    'Sources/Application/Models/',
                    'Tests/',
                    'Views/',
                    'Public/'
                ],
                'config_files': ['Package.swift', 'config.json']
            },
            'swiftui': {
                'language': LanguageType.SWIFT,
                'patterns': [
                    r'import\s+SwiftUI',
                    r'struct.*:\s*View',
                    r'struct.*:\s*App',
                    r'struct.*:\s*Scene',
                    r'struct.*:\s*PreviewProvider',
                    r'var\s+body:\s*some\s+View',
                    r'var\s+body:\s*some\s+Scene',
                    r'@main',
                    r'@State',
                    r'@StateObject',
                    r'@ObservedObject',
                    r'@EnvironmentObject',
                    r'@Environment',
                    r'@Binding',
                    r'@Published',
                    r'@FocusState',
                    r'@AppStorage',
                    r'@SceneStorage',
                    r'@FetchRequest',
                    r'@SectionedFetchRequest',
                    r'@GestureState',
                    r'@ScaledMetric',
                    r'@UIApplicationDelegateAdaptor',
                    r'@NSApplicationDelegateAdaptor',
                    r'Text\(',
                    r'Image\(',
                    r'Button\(',
                    r'TextField\(',
                    r'SecureField\(',
                    r'Toggle\(',
                    r'Slider\(',
                    r'Picker\(',
                    r'DatePicker\(',
                    r'ColorPicker\(',
                    r'ProgressView\(',
                    r'List\s*\{',
                    r'List\(',
                    r'ForEach\(',
                    r'ScrollView\s*\{',
                    r'ScrollView\(',
                    r'VStack\s*\{',
                    r'VStack\(',
                    r'HStack\s*\{',
                    r'HStack\(',
                    r'ZStack\s*\{',
                    r'ZStack\(',
                    r'NavigationView\s*\{',
                    r'NavigationView\(',
                    r'NavigationLink\(',
                    r'NavigationStack\s*\{',
                    r'NavigationStack\(',
                    r'TabView\s*\{',
                    r'TabView\(',
                    r'Form\s*\{',
                    r'Form\(',
                    r'Section\s*\{',
                    r'Section\(',
                    r'Group\s*\{',
                    r'Group\(',
                    r'GeometryReader\s*\{',
                    r'GeometryReader\(',
                    r'\.frame\(',
                    r'\.padding\(',
                    r'\.background\(',
                    r'\.foregroundColor\(',
                    r'\.font\(',
                    r'\.cornerRadius\(',
                    r'\.shadow\(',
                    r'\.overlay\(',
                    r'\.opacity\(',
                    r'\.scaleEffect\(',
                    r'\.rotationEffect\(',
                    r'\.animation\(',
                    r'\.transition\(',
                    r'\.onAppear\s*\{',
                    r'\.onAppear\(',
                    r'\.onDisappear\s*\{',
                    r'\.onDisappear\(',
                    r'\.onChange\(',
                    r'\.onTapGesture\s*\{',
                    r'\.onTapGesture\(',
                    r'\.sheet\(',
                    r'\.alert\(',
                    r'\.fullScreenCover\(',
                    r'\.popover\(',
                    r'\.confirmationDialog\(',
                    r'\.toolbar\s*\{',
                    r'\.toolbar\(',
                    r'\.navigationTitle\(',
                    r'\.navigationBarTitle\(',
                    r'\.tabItem\s*\{',
                    r'\.tabItem\(',
                    r'ContentView',
                    r'PreviewProvider',
                    r'\.preview',
                ],
                'files': [
                    'ContentView.swift',
                    'App.swift',
                    '*App.swift',
                    'Info.plist'
                ],
                'directories': [
                    'Views/',
                    'Models/',
                    'ViewModels/',
                    'Preview Content/',
                    'Assets.xcassets/'
                ],
                'config_files': ['*.xcodeproj', '*.xcworkspace', 'Package.swift']
            },
            
            # Kotlin Frameworks
            'ktor': {
                'language': LanguageType.KOTLIN,
                'patterns': [
                    r'import\s+io\.ktor\.',
                    r'import\s+io\.ktor\.application\.',
                    r'import\s+io\.ktor\.server\.',
                    r'import\s+io\.ktor\.client\.',
                    r'import\s+io\.ktor\.features\.',
                    r'import\s+io\.ktor\.routing\.',
                    r'import\s+io\.ktor\.http\.',
                    r'import\s+io\.ktor\.request\.',
                    r'import\s+io\.ktor\.response\.',
                    r'import\s+io\.ktor\.sessions\.',
                    r'import\s+io\.ktor\.auth\.',
                    r'import\s+io\.ktor\.locations\.',
                    r'import\s+io\.ktor\.websocket\.',
                    r'Application',
                    r'ApplicationCall',
                    r'PipelineContext',
                    r'embeddedServer\(',
                    r'fun\s+Application\.module',
                    r'fun\s+Application\.',
                    r'install\(',
                    r'routing\s*\{',
                    r'get\s*\{',
                    r'post\s*\{',
                    r'put\s*\{',
                    r'delete\s*\{',
                    r'patch\s*\{',
                    r'head\s*\{',
                    r'options\s*\{',
                    r'route\(',
                    r'call\.respond\(',
                    r'call\.respondText\(',
                    r'call\.respondHtml\(',
                    r'call\.respondFile\(',
                    r'call\.respondRedirect\(',
                    r'call\.receive\(',
                    r'call\.receiveText\(',
                    r'call\.receiveStream\(',
                    r'call\.parameters',
                    r'call\.request\.',
                    r'call\.response\.',
                    r'call\.sessions\.',
                    r'call\.authentication',
                    r'ContentNegotiation',
                    r'CallLogging',
                    r'Authentication',
                    r'Sessions',
                    r'Locations',
                    r'WebSockets',
                    r'StatusPages',
                    r'CORS',
                    r'Compression',
                    r'CachingHeaders',
                    r'ConditionalHeaders',
                    r'PartialContent',
                    r'AutoHeadResponse',
                    r'DoubleReceive',
                    r'feature\(',
                    r'intercept\(',
                    r'authenticate\s*\{',
                    r'session\s*\{',
                    r'webSocket\(',
                    r'HttpClient\s*\{',
                    r'client\.',
                    r'ktorVersion',
                    r'ktor\{',
                    r'implementation\("io\.ktor:',
                ],
                'files': [
                    'Application.kt',
                    'build.gradle.kts',
                    'build.gradle',
                    'settings.gradle.kts',
                    'settings.gradle',
                    'gradle.properties'
                ],
                'directories': [
                    'src/main/kotlin/',
                    'src/main/resources/',
                    'src/test/kotlin/',
                    'resources/templates/',
                    'resources/static/'
                ],
                'config_files': ['build.gradle.kts', 'application.conf', 'application.yaml']
            },
            'android-jetpack': {
                'language': LanguageType.KOTLIN,
                'patterns': [
                    r'import\s+androidx\.',
                    r'import\s+androidx\.compose\.',
                    r'import\s+androidx\.lifecycle\.',
                    r'import\s+androidx\.room\.',
                    r'import\s+androidx\.navigation\.',
                    r'import\s+androidx\.hilt\.',
                    r'import\s+androidx\.work\.',
                    r'import\s+androidx\.paging\.',
                    r'import\s+androidx\.datastore\.',
                    r'import\s+androidx\.camera\.',
                    r'import\s+androidx\.fragment\.',
                    r'import\s+androidx\.activity\.',
                    r'import\s+androidx\.recyclerview\.',
                    r'import\s+androidx\.viewpager2\.',
                    r'import\s+androidx\.constraintlayout\.',
                    r'import\s+android\.os\.Bundle',
                    r'import\s+android\.view\.',
                    r'import\s+android\.widget\.',
                    r'@Composable',
                    r'@Preview',
                    r'@OptIn',
                    r'@ExperimentalMaterial3Api',
                    r'@AndroidEntryPoint',
                    r'@HiltAndroidApp',
                    r'@Inject',
                    r'@Module',
                    r'@InstallIn',
                    r'@Provides',
                    r'@Singleton',
                    r'@ViewModelScoped',
                    r'@Entity',
                    r'@Dao',
                    r'@Database',
                    r'@Query',
                    r'@Insert',
                    r'@Update',
                    r'@Delete',
                    r'@PrimaryKey',
                    r'@ColumnInfo',
                    r'@ForeignKey',
                    r'@TypeConverter',
                    r'ViewModel\(',
                    r'AndroidViewModel\(',
                    r'LiveData<',
                    r'MutableLiveData<',
                    r'StateFlow<',
                    r'MutableStateFlow\(',
                    r'viewModelScope',
                    r'lifecycleScope',
                    r'rememberCoroutineScope',
                    r'LaunchedEffect',
                    r'DisposableEffect',
                    r'SideEffect',
                    r'remember\s*\{',
                    r'remember\(',
                    r'rememberSaveable',
                    r'mutableStateOf\(',
                    r'derivedStateOf\s*\{',
                    r'collectAsState\(',
                    r'observeAsState\(',
                    r'Scaffold\s*\{',
                    r'Scaffold\(',
                    r'TopAppBar\(',
                    r'BottomNavigation\s*\{',
                    r'NavigationBar\s*\{',
                    r'NavigationRail\s*\{',
                    r'LazyColumn\s*\{',
                    r'LazyColumn\(',
                    r'LazyRow\s*\{',
                    r'LazyRow\(',
                    r'Box\s*\{',
                    r'Box\(',
                    r'Row\s*\{',
                    r'Row\(',
                    r'Column\s*\{',
                    r'Column\(',
                    r'Surface\s*\{',
                    r'Surface\(',
                    r'Card\s*\{',
                    r'Card\(',
                    r'Text\(',
                    r'Button\(',
                    r'IconButton\(',
                    r'FloatingActionButton\(',
                    r'TextField\(',
                    r'OutlinedTextField\(',
                    r'Checkbox\(',
                    r'RadioButton\(',
                    r'Switch\(',
                    r'Slider\(',
                    r'Image\(',
                    r'Icon\(',
                    r'CircularProgressIndicator\(',
                    r'LinearProgressIndicator\(',
                    r'AlertDialog\(',
                    r'Dialog\(',
                    r'DropdownMenu\s*\{',
                    r'ModalBottomSheet\s*\{',
                    r'NavHost\(',
                    r'NavController',
                    r'rememberNavController\(',
                    r'navigate\(',
                    r'popBackStack\(',
                    r'composable\(',
                    r'navigation\(',
                    r'Activity',
                    r'Fragment',
                    r'onCreate\(',
                    r'onCreateView\(',
                    r'onViewCreated\(',
                    r'onStart\(',
                    r'onResume\(',
                    r'onPause\(',
                    r'onStop\(',
                    r'onDestroy\(',
                    r'setContent\s*\{',
                    r'setContentView\(',
                    r'findViewById\(',
                    r'viewBinding',
                    r'dataBinding',
                    r'R\.layout\.',
                    r'R\.id\.',
                    r'R\.string\.',
                    r'R\.drawable\.',
                    r'R\.color\.',
                    r'android\{',
                    r'implementation\("androidx\.',
                ],
                'files': [
                    'MainActivity.kt',
                    'build.gradle.kts',
                    'build.gradle',
                    'settings.gradle.kts',
                    'settings.gradle',
                    'AndroidManifest.xml',
                    'gradle.properties'
                ],
                'directories': [
                    'app/src/main/java/',
                    'app/src/main/kotlin/',
                    'app/src/main/res/',
                    'app/src/main/res/layout/',
                    'app/src/main/res/values/',
                    'app/src/main/res/drawable/',
                    'app/src/androidTest/',
                    'app/src/test/'
                ],
                'config_files': ['build.gradle.kts', 'AndroidManifest.xml', 'proguard-rules.pro']
            },
            'spring-boot-kotlin': {
                'language': LanguageType.KOTLIN,
                'patterns': [
                    r'import\s+org\.springframework\.',
                    r'import\s+org\.springframework\.boot\.',
                    r'import\s+org\.springframework\.web\.',
                    r'import\s+org\.springframework\.data\.',
                    r'import\s+org\.springframework\.security\.',
                    r'@SpringBootApplication',
                    r'@RestController',
                    r'@Controller',
                    r'@Service',
                    r'@Repository',
                    r'@Component',
                    r'@Configuration',
                    r'@Bean',
                    r'@Autowired',
                    r'@Value\(',
                    r'@RequestMapping\(',
                    r'@GetMapping\(',
                    r'@PostMapping\(',
                    r'@PutMapping\(',
                    r'@DeleteMapping\(',
                    r'@PatchMapping\(',
                    r'@PathVariable',
                    r'@RequestParam',
                    r'@RequestBody',
                    r'@ResponseBody',
                    r'@ResponseStatus',
                    r'@ExceptionHandler',
                    r'@ControllerAdvice',
                    r'@RestControllerAdvice',
                    r'@Valid',
                    r'@Validated',
                    r'@Entity',
                    r'@Table\(',
                    r'@Id',
                    r'@GeneratedValue',
                    r'@Column\(',
                    r'@OneToMany',
                    r'@ManyToOne',
                    r'@OneToOne',
                    r'@ManyToMany',
                    r'@JoinColumn',
                    r'@JoinTable',
                    r'@Query\(',
                    r'@Modifying',
                    r'@Transactional',
                    r'@EnableWebSecurity',
                    r'@EnableGlobalMethodSecurity',
                    r'@PreAuthorize',
                    r'@PostAuthorize',
                    r'@Secured',
                    r'@RolesAllowed',
                    r'@EnableScheduling',
                    r'@Scheduled\(',
                    r'@EnableAsync',
                    r'@Async',
                    r'@EnableCaching',
                    r'@Cacheable',
                    r'@CacheEvict',
                    r'@CachePut',
                    r'@EnableConfigurationProperties',
                    r'@ConfigurationProperties',
                    r'@ConditionalOnProperty',
                    r'@ConditionalOnClass',
                    r'@ConditionalOnMissingBean',
                    r'@Profile',
                    r'@ActiveProfiles',
                    r'@TestConfiguration',
                    r'@SpringBootTest',
                    r'@WebMvcTest',
                    r'@DataJpaTest',
                    r'@MockBean',
                    r'@SpyBean',
                    r'ResponseEntity',
                    r'HttpStatus',
                    r'JpaRepository',
                    r'CrudRepository',
                    r'PagingAndSortingRepository',
                    r'RestTemplate',
                    r'WebClient',
                    r'SecurityFilterChain',
                    r'PasswordEncoder',
                    r'AuthenticationManager',
                    r'UserDetailsService',
                    r'fun\s+main\(',
                    r'runApplication<',
                    r'SpringApplication\.run\(',
                    r'application\.properties',
                    r'application\.yml',
                    r'application\.yaml',
                    r'bootstrap\.properties',
                    r'bootstrap\.yml',
                    r'implementation\("org\.springframework\.boot:',
                ],
                'files': [
                    'Application.kt',
                    'build.gradle.kts',
                    'build.gradle',
                    'settings.gradle.kts',
                    'settings.gradle',
                    'application.properties',
                    'application.yml',
                    'application.yaml'
                ],
                'directories': [
                    'src/main/kotlin/',
                    'src/main/resources/',
                    'src/test/kotlin/',
                    'src/main/resources/templates/',
                    'src/main/resources/static/'
                ],
                'config_files': ['build.gradle.kts', 'application.properties', 'application.yml']
            },
            
            # Zig Frameworks
            'zap': {
                'language': LanguageType.ZIG,
                'patterns': [
                    r'const\s+zap\s*=\s*@import\("zap"\)',
                    r'zap\.HttpServer',
                    r'zap\.SimpleHttpServer',
                    r'zap\.SimpleEndpoint',
                    r'\.get\("',
                    r'\.post\("',
                ],
                'files': [],
                'config_files': ['build.zig']
            },
            'ziggy': {
                'language': LanguageType.ZIG,
                'patterns': [
                    r'const\s+ziggy\s*=\s*@import\("ziggy"\)',
                    r'ziggy\.Http',
                    r'ziggy\.Router',
                    r'ziggy\.Request',
                    r'ziggy\.Response',
                ],
                'files': [],
                'config_files': ['build.zig']
            },
            'mecha': {
                'language': LanguageType.ZIG,
                'patterns': [
                    r'const\s+mecha\s*=\s*@import\("mecha"\)',
                    r'mecha\.combine',
                    r'mecha\.oneOf',
                    r'mecha\.discard',
                    r'mecha\.ParseError',
                ],
                'files': [],
                'config_files': ['build.zig']
            },
            'zig-network': {
                'language': LanguageType.ZIG,
                'patterns': [
                    r'const\s+network\s*=\s*@import\("network"\)',
                    r'network\.Socket',
                    r'network\.Address',
                    r'network\.tcp',
                    r'network\.udp',
                ],
                'files': [],
                'config_files': ['build.zig']
            },
            'ziglings': {
                'language': LanguageType.ZIG,
                'patterns': [
                    r'const\s+ziglings\s*=\s*@import\("ziglings"\)',
                    r'// This is a Ziglings exercise',
                    r'// See https://github.com/ratfactor/ziglings',
                ],
                'files': ['exercises.zig'],
                'config_files': ['build.zig']
            },
            
            # Nim Frameworks
            'jester': {
                'language': LanguageType.NIM,
                'patterns': [
                    r'import\s+jester',
                    r'routes:',
                    r'resp\s+"',
                    r'@".*"\s*\(',
                    r'initJester\s*\(',
                ],
                'files': [],
                'config_files': ['nimble.json', 'nim.cfg']
            },
            'karax': {
                'language': LanguageType.NIM,
                'patterns': [
                    r'import\s+karax',
                    r'karax/karaxdsl',
                    r'karax/vdom',
                    r'karax/kdom',
                    r'proc\s+\w+Node\s*\(',
                    r'buildHtml:',
                ],
                'files': [],
                'config_files': ['nimble.json', 'nim.cfg']
            },
            'nimx': {
                'language': LanguageType.NIM,
                'patterns': [
                    r'import\s+nimx/',
                    r'nimx/window',
                    r'nimx/view',
                    r'nimx/timer',
                    r'nimx/animation',
                    r'newWindow\s*\(',
                ],
                'files': [],
                'config_files': ['nimble.json', 'nim.cfg']
            },
            'prologue': {
                'language': LanguageType.NIM,
                'patterns': [
                    r'import\s+prologue',
                    r'newApp\s*\(',
                    r'proc\s+\w+\s*\(ctx:\s*Context\)',
                    r'resp\s+\w+',
                    r'Prologue\s*\(',
                ],
                'files': [],
                'config_files': ['nimble.json', 'nim.cfg']
            },
            'norm': {
                'language': LanguageType.NIM,
                'patterns': [
                    r'import\s+norm/',
                    r'model\s+\w+',
                    r'proc\s+dbValue\s*\(',
                    r'init\s+\w+Db\s*\(',
                    r'type\s+\w+\s*=\s*ref\s+object\s+of\s+Model',
                ],
                'files': [],
                'config_files': ['nimble.json', 'nim.cfg']
            },
            
            # Crystal Frameworks
            'lucky': {
                'language': LanguageType.CRYSTAL,
                'patterns': [
                    r'require\s+"lucky"',
                    r'class\s+\w+\s+<\s+Lucky::Action',
                    r'Lucky::BaseAppServer',
                    r'Lucky::Env',
                    r'Lucky::Router',
                    r'Lucky::RouteHandler',
                ],
                'files': ['lucky.cr'],
                'config_files': ['shard.yml']
            },
            'amber': {
                'language': LanguageType.CRYSTAL,
                'patterns': [
                    r'require\s+"amber"',
                    r'Amber::Server',
                    r'Amber::Controller::Base',
                    r'Amber::Pipe',
                    r'Amber::Router',
                    r'Amber::Validator',
                ],
                'files': ['amber.cr'],
                'config_files': ['shard.yml']
            },
            'kemal': {
                'language': LanguageType.CRYSTAL,
                'patterns': [
                    r'require\s+"kemal"',
                    r'Kemal\.run',
                    r'Kemal\.config',
                    r'get\s+"\/\w*"\s+do',
                    r'post\s+"\/\w*"\s+do',
                    r'HTTP::Server::Context',
                ],
                'files': ['app.cr'],
                'config_files': ['shard.yml']
            },
            'granite': {
                'language': LanguageType.CRYSTAL,
                'patterns': [
                    r'require\s+"granite"',
                    r'class\s+\w+\s+<\s+Granite::Base',
                    r'Granite::Connections',
                    r'Granite::Query',
                    r'Granite::Columns',
                ],
                'files': [],
                'config_files': ['shard.yml']
            },
            'crecto': {
                'language': LanguageType.CRYSTAL,
                'patterns': [
                    r'require\s+"crecto"',
                    r'module\s+\w+Repo\s*;?\s*include\s+Crecto::Repo',
                    r'class\s+\w+\s+<\s+Crecto::Model',
                    r'Crecto::Changeset',
                    r'Crecto::Query',
                ],
                'files': [],
                'config_files': ['shard.yml']
            },
            
            # Haskell Frameworks
            'yesod': {
                'language': LanguageType.HASKELL,
                'patterns': [
                    r'import\s+Yesod',
                    r'import\s+Yesod\.\w+',
                    r'mkYesod\s+"\w+"',
                    r'mkMigrate\s+"\w+"',
                    r'instance\s+Yesod\s+\w+',
                    r'warp\s+\d+\s+\w+',
                ],
                'files': [],
                'config_files': ['package.yaml', 'stack.yaml', 'cabal.project']
            },
            'scotty': {
                'language': LanguageType.HASKELL,
                'patterns': [
                    r'import\s+Web\.Scotty',
                    r'import\s+Web\.Scotty\.\w+',
                    r'scotty\s+\d+',
                    r'ScottyM',
                    r'get\s+"[^"]+"',
                    r'post\s+"[^"]+"',
                ],
                'files': [],
                'config_files': ['package.yaml', 'stack.yaml', 'cabal.project']
            },
            'servant': {
                'language': LanguageType.HASKELL,
                'patterns': [
                    r'import\s+Servant',
                    r'import\s+Servant\.\w+',
                    r'type\s+API\s*=',
                    r'serve\s+\(Proxy\s*::',
                    r'::>\s+',
                    r'type\s+\w+API',
                ],
                'files': [],
                'config_files': ['package.yaml', 'stack.yaml', 'cabal.project']
            },
            'snap': {
                'language': LanguageType.HASKELL,
                'patterns': [
                    r'import\s+Snap',
                    r'import\s+Snap\.\w+',
                    r'Snap\s*\(',
                    r'makeSnaplet',
                    r'addRoutes',
                    r'SnapletInit',
                ],
                'files': [],
                'config_files': ['package.yaml', 'stack.yaml', 'cabal.project']
            },
            'reflex': {
                'language': LanguageType.HASKELL,
                'patterns': [
                    r'import\s+Reflex',
                    r'import\s+Reflex\.Dom',
                    r'MonadWidget',
                    r'mainWidget',
                    r'el\s+"\w+"',
                    r'Dynamic\s+t',
                ],
                'files': [],
                'config_files': ['package.yaml', 'stack.yaml', 'cabal.project']
            },
            
            # F# Frameworks
            'giraffe': {
                'language': LanguageType.FSHARP,
                'patterns': [
                    r'open\s+Giraffe',
                    r'open\s+Giraffe\.\w+',
                    r'webApp\s*{',
                    r'choose\s*\[',
                    r'route\s+"[^"]+"',
                    r'HttpHandler',
                ],
                'files': [],
                'config_files': ['paket.dependencies', 'fsproj', 'fsharplint.json']
            },
            'saturn': {
                'language': LanguageType.FSHARP,
                'patterns': [
                    r'open\s+Saturn',
                    r'open\s+Saturn\.\w+',
                    r'application\s*{',
                    r'router\s*{',
                    r'Controller\.\w+',
                    r'endpoint\s*{',
                ],
                'files': [],
                'config_files': ['paket.dependencies', 'fsproj', 'fsharplint.json']
            },
            'falco': {
                'language': LanguageType.FSHARP,
                'patterns': [
                    r'open\s+Falco',
                    r'open\s+Falco\.\w+',
                    r'webHost\s*{',
                    r'Response\.\w+',
                    r'Request\.\w+',
                    r'Route\.\w+',
                ],
                'files': [],
                'config_files': ['paket.dependencies', 'fsproj', 'fsharplint.json']
            },
            'fable': {
                'language': LanguageType.FSHARP,
                'patterns': [
                    r'open\s+Fable',
                    r'open\s+Fable\.\w+',
                    r'importAll\s+"[^"]+"',
                    r'importMember\s+"[^"]+"',
                    r'\[<Global>\]',
                    r'\[<Emit\([^)]+\)>\]',
                ],
                'files': [],
                'config_files': ['paket.dependencies', 'fsproj', 'webpack.config.js']
            },
            'elmish': {
                'language': LanguageType.FSHARP,
                'patterns': [
                    r'open\s+Elmish',
                    r'open\s+Elmish\.\w+',
                    r'Program\.mkProgram',
                    r'Program\.mkSimple',
                    r'\|>\s*Program\.withReactSynchronous',
                    r'type\s+Msg\s*=',
                    r'type\s+Model\s*=',
                ],
                'files': [],
                'config_files': ['paket.dependencies', 'fsproj', 'webpack.config.js']
            },
            
            # Erlang Frameworks
            'cowboy': {
                'language': LanguageType.ERLANG,
                'patterns': [
                    r'-include_lib\("cowboy/.*"\)',
                    r'cowboy:start_clear',
                    r'cowboy:start_tls',
                    r'cowboy_router',
                    r'cowboy_req:',
                    r':cowboy_handler',
                ],
                'files': [],
                'config_files': ['rebar.config', 'erlang.mk']
            },
            'otp': {
                'language': LanguageType.ERLANG,
                'patterns': [
                    r'-behaviour\(gen_server\)',
                    r'-behaviour\(supervisor\)',
                    r'gen_server:start_link',
                    r'gen_server:call',
                    r'handle_call\(\w+,\s*From,\s*State\)',
                    r'handle_cast\(\w+,\s*State\)',
                ],
                'files': [],
                'config_files': ['rebar.config', 'erlang.mk']
            },
            'phoenix': {
                'language': LanguageType.ERLANG,
                'patterns': [
                    r'use\s+Phoenix\.\w+',
                    r'Phoenix\.Router',
                    r'Phoenix\.Controller',
                    r'Phoenix\.Endpoint',
                    r'Phoenix\.Channel',
                    r'Phoenix\.Presence',
                ],
                'files': ['mix.exs'],
                'config_files': ['config.exs']
            },
            'chicago_boss': {
                'language': LanguageType.ERLANG,
                'patterns': [
                    r'-compile\(\[boss_db\]\)',
                    r'boss\.config',
                    r'boss_db:find',
                    r'boss_mail',
                    r'boss_news',
                    r'boss_web',
                ],
                'files': [],
                'config_files': ['boss.config']
            },
            'ejabberd': {
                'language': LanguageType.ERLANG,
                'patterns': [
                    r'-include\("ejabberd\.hrl"\)',
                    r'-include\("logger\.hrl"\)',
                    r'ejabberd_commands',
                    r'ejabberd_config',
                    r'ejabberd_router',
                    r'gen_mod',
                ],
                'files': [],
                'config_files': ['ejabberd.yml']
            },
            
            # R Frameworks
            'shiny': {
                'language': LanguageType.R,
                'patterns': [
                    r'library\(shiny\)',
                    r'require\(shiny\)',
                    r'shinyApp\(',
                    r'renderPlot\(',
                    r'renderTable\(',
                    r'fluidPage\(',
                    r'ui <- fluidPage',
                    r'server <- function\(input, output',
                ],
                'files': ['app.R', 'ui.R', 'server.R'],
                'config_files': ['DESCRIPTION']
            },
            'plumber': {
                'language': LanguageType.R,
                'patterns': [
                    r'library\(plumber\)',
                    r'require\(plumber\)',
                    r'#\* @get',
                    r'#\* @post',
                    r'#\* @apiTitle',
                    r'plumb\(',
                    r'pr\(\)',
                ],
                'files': ['plumber.R'],
                'config_files': ['DESCRIPTION']
            },
            'tidyverse': {
                'language': LanguageType.R,
                'patterns': [
                    r'library\(tidyverse\)',
                    r'require\(tidyverse\)',
                    r'library\(dplyr\)',
                    r'library\(ggplot2\)',
                    r'library\(tidyr\)',
                    r'library\(purrr\)',
                    r'%>%',
                    r'mutate\(',
                    r'filter\(',
                    r'select\(',
                    r'group_by\(',
                ],
                'files': [],
                'config_files': ['DESCRIPTION']
            },
            'rmarkdown': {
                'language': LanguageType.R,
                'patterns': [
                    r'library\(rmarkdown\)',
                    r'require\(rmarkdown\)',
                    r'```\{r',
                    r'knitr::opts_chunk',
                    r'render\(',
                ],
                'files': ['*.Rmd'],
                'config_files': ['DESCRIPTION']
            },
            'testthat': {
                'language': LanguageType.R,
                'patterns': [
                    r'library\(testthat\)',
                    r'require\(testthat\)',
                    r'test_that\(',
                    r'expect_equal\(',
                    r'expect_true\(',
                    r'expect_error\(',
                ],
                'files': ['testthat.R'],
                'config_files': ['DESCRIPTION']
            },
            
            # Julia Frameworks
            'genie': {
                'language': LanguageType.JULIA,
                'patterns': [
                    r'using\s+Genie',
                    r'import\s+Genie',
                    r'Genie\.render',
                    r'Genie\.Router',
                    r'route\("[^"]+"',
                    r'@get\s+"',
                    r'@post\s+"',
                ],
                'files': ['routes.jl'],
                'config_files': ['Project.toml', 'Manifest.toml']
            },
            'franklin': {
                'language': LanguageType.JULIA,
                'patterns': [
                    r'using\s+Franklin',
                    r'import\s+Franklin',
                    r'@def\s+',
                    r'\{\{[^}]+\}\}',
                    r'\{\{fill[^}]+\}\}',
                ],
                'files': ['_layout', 'index.md'],
                'config_files': ['Project.toml', 'Manifest.toml']
            },
            'pluto': {
                'language': LanguageType.JULIA,
                'patterns': [
                    r'using\s+Pluto',
                    r'import\s+Pluto',
                    r'Pluto\.run',
                    r'# \^\^\^',
                    r'html"',
                    r'md"',
                ],
                'files': ['*.jl'],
                'config_files': ['Project.toml', 'Manifest.toml']
            },
            'flux': {
                'language': LanguageType.JULIA,
                'patterns': [
                    r'using\s+Flux',
                    r'import\s+Flux',
                    r'Flux\.\w+',
                    r'Dense\(',
                    r'Chain\(',
                    r'Flux\.train!',
                    r'Flux\.params',
                ],
                'files': [],
                'config_files': ['Project.toml', 'Manifest.toml']
            },
            'jumpjl': {
                'language': LanguageType.JULIA,
                'patterns': [
                    r'using\s+JuMP',
                    r'import\s+JuMP',
                    r'Model\(',
                    r'@variable',
                    r'@constraint',
                    r'@objective',
                    r'optimize!',
                ],
                'files': [],
                'config_files': ['Project.toml', 'Manifest.toml']
            },
            
            # Terraform Provider Patterns
            'terraform_aws': {
                'language': LanguageType.TERRAFORM,
                'patterns': [
                    r'provider\s+"aws"',
                    r'resource\s+"aws_',
                    r'data\s+"aws_',
                    r'aws_vpc',
                    r'aws_instance',
                    r'aws_s3_bucket',
                    r'aws_lambda_function',
                ],
                'files': [],
                'config_files': ['terraform.tfvars', 'terraform.tfstate']
            },
            'terraform_azure': {
                'language': LanguageType.TERRAFORM,
                'patterns': [
                    r'provider\s+"azurerm"',
                    r'resource\s+"azurerm_',
                    r'data\s+"azurerm_',
                    r'azurerm_virtual_machine',
                    r'azurerm_resource_group',
                    r'azurerm_app_service',
                    r'azurerm_storage_account',
                ],
                'files': [],
                'config_files': ['terraform.tfvars', 'terraform.tfstate']
            },
            'terraform_gcp': {
                'language': LanguageType.TERRAFORM,
                'patterns': [
                    r'provider\s+"google"',
                    r'resource\s+"google_',
                    r'data\s+"google_',
                    r'google_compute_instance',
                    r'google_storage_bucket',
                    r'google_container_cluster',
                    r'google_cloud_run_service',
                ],
                'files': [],
                'config_files': ['terraform.tfvars', 'terraform.tfstate']
            },
            'terraform_kubernetes': {
                'language': LanguageType.TERRAFORM,
                'patterns': [
                    r'provider\s+"kubernetes"',
                    r'resource\s+"kubernetes_',
                    r'data\s+"kubernetes_',
                    r'kubernetes_namespace',
                    r'kubernetes_deployment',
                    r'kubernetes_service',
                    r'kubernetes_pod',
                ],
                'files': [],
                'config_files': ['terraform.tfvars', 'terraform.tfstate']
            },
            'terraform_modules': {
                'language': LanguageType.TERRAFORM,
                'patterns': [
                    r'module\s+"\w+"\s+{',
                    r'source\s+=\s+"',
                    r'variable\s+"\w+"\s+{',
                    r'output\s+"\w+"\s+{',
                    r'terraform\s+{',
                    r'backend\s+"\w+"\s+{',
                ],
                'files': [],
                'config_files': ['terraform.tfvars', 'terraform.tfstate']
            },
            
            # Ansible Module Patterns
            'ansible_apt': {
                'language': LanguageType.ANSIBLE,
                'patterns': [
                    r'\s+apt:',
                    r'\s+name:\s+\w+',
                    r'\s+state:\s+(present|latest|absent)',
                    r'\s+update_cache:\s+(yes|no)',
                    r'become:\s+true',
                ],
                'files': ['playbook.yml', 'site.yml'],
                'config_files': ['ansible.cfg']
            },
            'ansible_yum': {
                'language': LanguageType.ANSIBLE,
                'patterns': [
                    r'\s+yum:',
                    r'\s+name:\s+\w+',
                    r'\s+state:\s+(present|latest|absent)',
                    r'\s+enablerepo:\s+\w+',
                    r'become:\s+true',
                ],
                'files': ['playbook.yml', 'site.yml'],
                'config_files': ['ansible.cfg']
            },
            'ansible_service': {
                'language': LanguageType.ANSIBLE,
                'patterns': [
                    r'\s+service:',
                    r'\s+name:\s+\w+',
                    r'\s+state:\s+(started|stopped|restarted|reloaded)',
                    r'\s+enabled:\s+(yes|no)',
                    r'become:\s+true',
                ],
                'files': ['playbook.yml', 'site.yml'],
                'config_files': ['ansible.cfg']
            },
            'ansible_file': {
                'language': LanguageType.ANSIBLE,
                'patterns': [
                    r'\s+file:',
                    r'\s+path:\s+.+',
                    r'\s+state:\s+(directory|file|touch|absent|link)',
                    r'\s+mode:\s+["\']?[0-7]{3,4}["\']?',
                    r'\s+owner:\s+\w+',
                    r'\s+group:\s+\w+',
                ],
                'files': ['playbook.yml', 'site.yml'],
                'config_files': ['ansible.cfg']
            },
            'ansible_git': {
                'language': LanguageType.ANSIBLE,
                'patterns': [
                    r'\s+git:',
                    r'\s+repo:\s+.+',
                    r'\s+dest:\s+.+',
                    r'\s+version:\s+.+',
                    r'\s+clone:\s+(yes|no)',
                    r'\s+update:\s+(yes|no)',
                ],
                'files': ['playbook.yml', 'site.yml'],
                'config_files': ['ansible.cfg']
            }
        }

    def _init_file_extension_map(self) -> Dict[str, LanguageType]:
        """Initialize file extension to language mapping."""
        return {
            '.py': LanguageType.PYTHON,
            '.js': LanguageType.JAVASCRIPT,
            '.jsx': LanguageType.JAVASCRIPT,
            '.ts': LanguageType.TYPESCRIPT,
            '.tsx': LanguageType.TYPESCRIPT,
            '.java': LanguageType.JAVA,
            '.go': LanguageType.GO,
            '.rs': LanguageType.RUST,
            '.swift': LanguageType.SWIFT,
            '.kt': LanguageType.KOTLIN,
            '.cs': LanguageType.CSHARP,
            '.rb': LanguageType.RUBY,
            '.php': LanguageType.PHP,
            '.scala': LanguageType.SCALA,
            '.ex': LanguageType.ELIXIR,
            '.exs': LanguageType.ELIXIR,
            '.clj': LanguageType.CLOJURE,
            '.cljs': LanguageType.CLOJURE,
            '.cpp': LanguageType.CPP,
            '.cc': LanguageType.CPP,
            '.cxx': LanguageType.CPP,
            '.c': LanguageType.C,
            '.h': LanguageType.C,
            '.hpp': LanguageType.CPP,
            # Additional languages from Phase 12.A
            '.zig': LanguageType.ZIG,
            '.nim': LanguageType.NIM,
            '.nims': LanguageType.NIM,
            '.cr': LanguageType.CRYSTAL,
            '.hs': LanguageType.HASKELL,
            '.lhs': LanguageType.HASKELL,
            '.fs': LanguageType.FSHARP,
            '.fsi': LanguageType.FSHARP,
            '.fsx': LanguageType.FSHARP,
            '.erl': LanguageType.ERLANG,
            '.hrl': LanguageType.ERLANG,
            '.sql': LanguageType.SQL,
            '.ddl': LanguageType.SQL,
            '.dml': LanguageType.SQL,
            '.sh': LanguageType.BASH,
            '.bash': LanguageType.BASH,
            '.ps1': LanguageType.POWERSHELL,
            '.psm1': LanguageType.POWERSHELL,
            '.psd1': LanguageType.POWERSHELL,
            '.lua': LanguageType.LUA,
            '.r': LanguageType.R,
            '.R': LanguageType.R,
            '.m': LanguageType.MATLAB,
            '.mat': LanguageType.MATLAB,
            '.jl': LanguageType.JULIA,
            '.tf': LanguageType.TERRAFORM,
            '.tfvars': LanguageType.TERRAFORM,
            '.yml': LanguageType.YAML,
            '.yaml': LanguageType.YAML,
            '.json': LanguageType.JSON,
            '.jsonc': LanguageType.JSON,
        }

    def detect_language_and_frameworks(self, 
                                     file_path: Optional[str] = None,
                                     source_code: Optional[str] = None,
                                     project_root: Optional[str] = None) -> LanguageInfo:
        """
        Detect programming language and frameworks from file or source code.
        
        Args:
            file_path: Path to the source file
            source_code: Source code content
            project_root: Root directory of the project
            
        Returns:
            LanguageInfo with detected language and frameworks
        """
        # First, try to detect language from file extension
        detected_language = LanguageType.UNKNOWN
        confidence = 0.0
        
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in self.file_extension_map:
                detected_language = self.file_extension_map[file_ext]
                confidence = 0.8  # High confidence from file extension
        
        # If we have source code, analyze it for language patterns
        if source_code and detected_language != LanguageType.UNKNOWN:
            # Validate and refine language detection using content analysis
            content_confidence = self._analyze_language_content(source_code, detected_language)
            confidence = max(confidence, content_confidence)
        elif source_code:
            # Try to detect language from content alone
            detected_language, confidence = self._detect_language_from_content(source_code)
        
        # Detect frameworks
        frameworks = []
        if source_code:
            frameworks = self._detect_frameworks(source_code, detected_language, project_root)
        
        # Get language features
        language_features = self._get_language_features(detected_language, source_code)
        
        # Get file patterns for this language
        file_patterns = self._get_file_patterns(detected_language)
        
        return LanguageInfo(
            language=detected_language,
            confidence=confidence,
            frameworks=frameworks,
            file_patterns=file_patterns,
            language_features=language_features
        )

    def _analyze_language_content(self, source_code: str, suspected_language: LanguageType) -> float:
        """
        Analyze source code content to validate language detection.
        
        Args:
            source_code: Source code to analyze
            suspected_language: Language detected from file extension
            
        Returns:
            Confidence score for the language detection
        """
        if suspected_language not in self.language_patterns:
            return 0.0
        
        patterns = self.language_patterns[suspected_language]
        total_score = 0
        max_score = 0
        
        # Check syntax patterns
        for pattern in patterns.get('syntax', []):
            max_score += 1
            if re.search(pattern, source_code, re.MULTILINE):
                total_score += 1
        
        # Check import patterns
        for pattern in patterns.get('imports', []):
            max_score += 1
            if re.search(pattern, source_code, re.MULTILINE):
                total_score += 1
        
        # Check for keywords
        keywords = patterns.get('keywords', [])
        if keywords:
            max_score += len(keywords)
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', source_code):
                    total_score += 1
        
        return total_score / max_score if max_score > 0 else 0.0

    def _detect_language_from_content(self, source_code: str) -> Tuple[LanguageType, float]:
        """
        Detect language from source code content alone.
        
        Args:
            source_code: Source code to analyze
            
        Returns:
            Tuple of (detected_language, confidence)
        """
        best_language = LanguageType.UNKNOWN
        best_confidence = 0.0
        
        for language, patterns in self.language_patterns.items():
            confidence = self._analyze_language_content(source_code, language)
            if confidence > best_confidence:
                best_confidence = confidence
                best_language = language
        
        return best_language, best_confidence

    def _detect_frameworks(self, 
                          source_code: str, 
                          language: LanguageType,
                          project_root: Optional[str] = None) -> List[FrameworkInfo]:
        """
        Detect frameworks being used in the source code.
        
        Args:
            source_code: Source code to analyze
            language: Detected programming language
            project_root: Root directory of the project
            
        Returns:
            List of detected frameworks
        """
        frameworks = []
        
        for framework_name, framework_config in self.framework_patterns.items():
            # Skip if framework doesn't match the detected language
            if framework_config['language'] != language:
                continue
            
            confidence = 0.0
            indicators = []
            
            # Check source code patterns
            patterns = framework_config.get('patterns', [])
            pattern_matches = 0
            for pattern in patterns:
                if re.search(pattern, source_code, re.MULTILINE | re.IGNORECASE):
                    pattern_matches += 1
                    indicators.append(f"Pattern: {pattern}")
            
            if patterns:
                confidence += (pattern_matches / len(patterns)) * 0.7
            
            # Check for framework-specific files in project
            if project_root:
                file_matches = 0
                files = framework_config.get('files', [])
                for file_name in files:
                    file_path = Path(project_root) / file_name
                    if file_path.exists():
                        file_matches += 1
                        indicators.append(f"File: {file_name}")
                
                if files:
                    confidence += (file_matches / len(files)) * 0.3
            
            # Only include frameworks with reasonable confidence
            if confidence > 0.2:
                frameworks.append(FrameworkInfo(
                    name=framework_name,
                    language=language,
                    confidence=confidence,
                    indicators=indicators
                ))
        
        # Sort by confidence
        frameworks.sort(key=lambda f: f.confidence, reverse=True)
        return frameworks

    def _get_language_features(self, 
                              language: LanguageType, 
                              source_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Get language-specific features and characteristics.
        
        Args:
            language: Detected language
            source_code: Source code to analyze
            
        Returns:
            Dictionary of language features
        """
        if language not in self.language_patterns:
            return {}
        
        patterns = self.language_patterns[language]
        features = {
            'comment_style': patterns.get('comment_style', '#'),
            'indent_style': patterns.get('indent_style', 'spaces'),
            'typical_indent': patterns.get('typical_indent', 4),
            'keywords': patterns.get('keywords', [])
        }
        
        # Analyze actual indentation if source code is provided
        if source_code:
            features['detected_indent'] = self._detect_indentation(source_code)
        
        return features

    def _detect_indentation(self, source_code: str) -> Dict[str, Any]:
        """
        Detect indentation style and size from source code.
        
        Args:
            source_code: Source code to analyze
            
        Returns:
            Dictionary with indentation information
        """
        lines = source_code.split('\n')
        space_indents = []
        tab_indents = []
        
        for line in lines:
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip(' '))
                leading_tabs = len(line) - len(line.lstrip('\t'))
                
                if leading_spaces > 0:
                    space_indents.append(leading_spaces)
                elif leading_tabs > 0:
                    tab_indents.append(leading_tabs)
        
        indent_info = {
            'uses_spaces': len(space_indents) > 0,
            'uses_tabs': len(tab_indents) > 0,
            'space_count': len(space_indents),
            'tab_count': len(tab_indents)
        }
        
        if space_indents:
            # Find most common indentation size
            from collections import Counter
            indent_sizes = [indent for indent in space_indents if indent > 0]
            if indent_sizes:
                most_common = Counter(indent_sizes).most_common(1)[0][0]
                indent_info['typical_spaces'] = most_common
        
        return indent_info

    def _get_file_patterns(self, language: LanguageType) -> List[str]:
        """
        Get typical file patterns for a language.
        
        Args:
            language: Programming language
            
        Returns:
            List of file patterns
        """
        patterns_map = {
            LanguageType.PYTHON: ['*.py', '*.pyw'],
            LanguageType.JAVASCRIPT: ['*.js', '*.jsx'],
            LanguageType.TYPESCRIPT: ['*.ts', '*.tsx'],
            LanguageType.JAVA: ['*.java'],
            LanguageType.GO: ['*.go'],
            LanguageType.RUST: ['*.rs'],
            LanguageType.SWIFT: ['*.swift'],
            LanguageType.KOTLIN: ['*.kt', '*.kts'],
            LanguageType.CSHARP: ['*.cs'],
            LanguageType.RUBY: ['*.rb'],
            LanguageType.PHP: ['*.php'],
            LanguageType.SCALA: ['*.scala'],
            LanguageType.ELIXIR: ['*.ex', '*.exs'],
            LanguageType.CLOJURE: ['*.clj', '*.cljs'],
            LanguageType.CPP: ['*.cpp', '*.cc', '*.cxx', '*.hpp'],
            LanguageType.C: ['*.c', '*.h'],
            # Additional languages from Phase 12.A
            LanguageType.ZIG: ['*.zig'],
            LanguageType.NIM: ['*.nim', '*.nims'],
            LanguageType.CRYSTAL: ['*.cr'],
            LanguageType.HASKELL: ['*.hs', '*.lhs'],
            LanguageType.FSHARP: ['*.fs', '*.fsi', '*.fsx'],
            LanguageType.ERLANG: ['*.erl', '*.hrl'],
            LanguageType.SQL: ['*.sql', '*.ddl', '*.dml'],
            LanguageType.BASH: ['*.sh', '*.bash'],
            LanguageType.POWERSHELL: ['*.ps1', '*.psm1', '*.psd1'],
            LanguageType.LUA: ['*.lua'],
            LanguageType.R: ['*.r', '*.R'],
            LanguageType.MATLAB: ['*.m', '*.mat'],
            LanguageType.JULIA: ['*.jl'],
            LanguageType.TERRAFORM: ['*.tf', '*.tfvars'],
            LanguageType.ANSIBLE: ['*.yml', '*.yaml', 'playbook.yml', 'site.yml'],
            LanguageType.YAML: ['*.yml', '*.yaml'],
            LanguageType.JSON: ['*.json', '*.jsonc'],
            LanguageType.DOCKERFILE: ['Dockerfile', 'Dockerfile.*', '*.dockerfile'],
        }
        
        return patterns_map.get(language, [])

    def get_llm_context_for_language(self, language_info: LanguageInfo) -> Dict[str, Any]:
        """
        Generate LLM context information for the detected language and frameworks.
        
        Args:
            language_info: Detected language information
            
        Returns:
            Context dictionary for LLM prompts
        """
        context = {
            'language': language_info.language.value,
            'confidence': language_info.confidence,
            'features': language_info.language_features,
            'file_patterns': language_info.file_patterns,
            'frameworks': []
        }
        
        # Add framework information
        for framework in language_info.frameworks:
            framework_info = {
                'name': framework.name,
                'confidence': framework.confidence,
                'indicators': framework.indicators,
                'guidance': self._get_framework_guidance(framework.name)
            }
            context['frameworks'].append(framework_info)
        
        # Add language-specific guidance for LLMs
        language_guidance = self._get_language_guidance(language_info.language)
        context['llm_guidance'] = language_guidance
        
        return context

    def _get_framework_guidance(self, framework_name: str) -> Dict[str, Any]:
        """
        Get framework-specific guidance for LLM patch generation.
        
        Args:
            framework_name: Name of the framework
            
        Returns:
            Guidance dictionary for the framework
        """
        framework_guidance = {
            # Python Frameworks
            'django': {
                'conventions': 'Django MTV architecture style',
                'patterns': [
                    'Use class-based views for complex logic',
                    'Follow Django naming conventions (snake_case)',
                    'Separate business logic from views',
                    'Use Django ORM for database operations',
                    'Implement proper model relationships'
                ],
                'imports': 'Use explicit imports from django modules',
                'best_practices': 'Follow Django\'s DRY principle, use built-in features like forms, authentication, and admin'
            },
            'flask': {
                'conventions': 'Flask microframework style',
                'patterns': [
                    'Use blueprints for organizing large applications',
                    'Follow Flask factory pattern for app creation',
                    'Use Flask-SQLAlchemy for ORM operations',
                    'Implement proper request/response handling'
                ],
                'imports': 'Use from flask import ... for core imports',
                'best_practices': 'Keep it simple, use extensions wisely, implement proper error handlers'
            },
            'fastapi': {
                'conventions': 'FastAPI async-first style',
                'patterns': [
                    'Use type hints for automatic validation',
                    'Implement async/await for I/O operations',
                    'Use Pydantic models for data validation',
                    'Follow RESTful API design principles'
                ],
                'imports': 'Use from fastapi import FastAPI, HTTPException, Depends',
                'best_practices': 'Leverage automatic API documentation, use dependency injection, implement proper async patterns'
            },
            'pyramid': {
                'conventions': 'Pyramid configuration-based style',
                'patterns': [
                    'Use configurator for app setup',
                    'Follow traversal or URL dispatch patterns',
                    'Implement views as callables',
                    'Use renderers for response formatting'
                ],
                'imports': 'Use from pyramid.config import Configurator',
                'best_practices': 'Choose appropriate URL mapping strategy, use authentication/authorization policies'
            },
            'tornado': {
                'conventions': 'Tornado async web server style',
                'patterns': [
                    'Use RequestHandler classes for endpoints',
                    'Implement async methods with coroutines',
                    'Handle WebSocket connections properly',
                    'Use IOLoop for async operations'
                ],
                'imports': 'Use import tornado.web, tornado.ioloop',
                'best_practices': 'Leverage non-blocking I/O, handle connection cleanup, use proper async patterns'
            },
            
            # JavaScript/TypeScript Frameworks
            'express': {
                'conventions': 'Express.js middleware-based style',
                'patterns': [
                    'Use middleware for cross-cutting concerns',
                    'Implement proper error handling middleware',
                    'Follow REST conventions for routes',
                    'Use router for modular route organization'
                ],
                'imports': 'Use const express = require("express") or ES6 imports',
                'best_practices': 'Order middleware correctly, handle async errors, use proper HTTP status codes'
            },
            'koa': {
                'conventions': 'Koa.js async-first middleware style',
                'patterns': [
                    'Use async/await in middleware',
                    'Leverage context (ctx) object properly',
                    'Implement error handling with try/catch',
                    'Use compose for middleware composition'
                ],
                'imports': 'Use const Koa = require("koa") or ES6 imports',
                'best_practices': 'Embrace async/await patterns, use proper middleware composition, handle errors gracefully'
            },
            'hapi': {
                'conventions': 'Hapi.js plugin-based architecture',
                'patterns': [
                    'Use plugins for feature organization',
                    'Implement proper route configuration',
                    'Use Joi for input validation',
                    'Follow server.route() patterns'
                ],
                'imports': 'Use const Hapi = require("@hapi/hapi")',
                'best_practices': 'Leverage built-in validation, use plugins effectively, implement proper authentication'
            },
            'fastify': {
                'conventions': 'Fastify high-performance style',
                'patterns': [
                    'Use schema validation for performance',
                    'Implement plugins with proper encapsulation',
                    'Use hooks for lifecycle management',
                    'Follow async/await patterns'
                ],
                'imports': 'Use const fastify = require("fastify")()',
                'best_practices': 'Leverage schema validation, use logging effectively, optimize for performance'
            },
            'nestjs': {
                'conventions': 'NestJS enterprise architecture style',
                'patterns': [
                    'Use decorators for metadata',
                    'Follow modular architecture with modules',
                    'Implement dependency injection properly',
                    'Use providers for services'
                ],
                'imports': 'Use import { Module, Controller, Injectable } from "@nestjs/common"',
                'best_practices': 'Follow SOLID principles, use TypeScript features, implement proper testing'
            },
            'angular': {
                'conventions': 'Angular component-based architecture',
                'patterns': [
                    'Use components for UI building blocks',
                    'Implement services for business logic',
                    'Use RxJS for reactive programming',
                    'Follow Angular style guide'
                ],
                'imports': 'Use import { Component, Injectable } from "@angular/core"',
                'best_practices': 'Use OnPush change detection, implement proper lifecycle hooks, avoid memory leaks with unsubscribe'
            },
            'react': {
                'conventions': 'React component-based UI style',
                'patterns': [
                    'Use functional components with hooks',
                    'Implement proper state management',
                    'Follow composition over inheritance',
                    'Use key props in lists'
                ],
                'imports': 'Use import React, { useState, useEffect } from "react"',
                'best_practices': 'Avoid unnecessary re-renders, use proper effect dependencies, implement error boundaries'
            },
            'vue': {
                'conventions': 'Vue.js reactive component style',
                'patterns': [
                    'Use Composition API for complex logic',
                    'Implement proper reactivity patterns',
                    'Follow single-file component structure',
                    'Use computed properties effectively'
                ],
                'imports': 'Use import { ref, reactive, computed } from "vue"',
                'best_practices': 'Leverage Vue reactivity system, use proper component communication, avoid mutating props'
            },
            
            # Java Frameworks
            'spring': {
                'conventions': 'Spring dependency injection style',
                'patterns': [
                    'Use annotations for configuration',
                    'Follow layered architecture (Controller, Service, Repository)',
                    'Implement proper exception handling',
                    'Use Spring profiles for environments'
                ],
                'imports': 'Use org.springframework imports',
                'best_practices': 'Leverage Spring Boot auto-configuration, use proper transaction management, implement AOP for cross-cutting concerns'
            },
            'spring-boot': {
                'conventions': 'Spring Boot convention-over-configuration style',
                'patterns': [
                    'Use @SpringBootApplication for main class',
                    'Follow REST controller patterns',
                    'Implement proper configuration with properties',
                    'Use starter dependencies'
                ],
                'imports': 'Use org.springframework.boot imports',
                'best_practices': 'Follow 12-factor app principles, use actuator for monitoring, implement proper logging'
            },
            'hibernate': {
                'conventions': 'Hibernate ORM mapping style',
                'patterns': [
                    'Use JPA annotations for entity mapping',
                    'Implement proper entity relationships',
                    'Follow repository pattern',
                    'Use HQL/JPQL for queries'
                ],
                'imports': 'Use javax.persistence and org.hibernate imports',
                'best_practices': 'Avoid N+1 queries, use proper fetching strategies, implement second-level cache wisely'
            },
            'struts': {
                'conventions': 'Struts MVC framework style',
                'patterns': [
                    'Use Action classes for controllers',
                    'Implement ActionForm for form handling',
                    'Follow struts-config.xml patterns',
                    'Use Struts tags in JSP'
                ],
                'imports': 'Use org.apache.struts imports',
                'best_practices': 'Validate input properly, use tiles for layout, implement proper error handling'
            },
            'play': {
                'conventions': 'Play Framework reactive style',
                'patterns': [
                    'Use controllers with async actions',
                    'Implement routes file properly',
                    'Follow RESTful conventions',
                    'Use Twirl templates for views'
                ],
                'imports': 'Use play.mvc and play.api imports',
                'best_practices': 'Leverage async capabilities, use proper error handling, implement caching strategies'
            },
            
            # C++ Frameworks
            'qt': {
                'conventions': 'Qt object-oriented style with signals/slots',
                'patterns': [
                    'Use Q_OBJECT macro for QObject classes',
                    'Implement signals and slots properly',
                    'Follow Qt naming conventions',
                    'Use Qt containers when appropriate'
                ],
                'imports': 'Use #include <QWidget>, <QObject>, etc.',
                'best_practices': 'Manage memory with parent-child relationships, use Qt\'s event system, avoid blocking the GUI thread'
            },
            'boost': {
                'conventions': 'Boost modern C++ library style',
                'patterns': [
                    'Use appropriate Boost libraries',
                    'Follow RAII principles',
                    'Implement proper exception safety',
                    'Use smart pointers for memory management'
                ],
                'imports': 'Use #include <boost/...> for specific libraries',
                'best_practices': 'Choose header-only libraries when possible, understand compile-time overhead, use Boost.Test for testing'
            },
            'poco': {
                'conventions': 'POCO C++ framework style',
                'patterns': [
                    'Use Poco namespaces appropriately',
                    'Implement proper error handling',
                    'Follow POCO threading patterns',
                    'Use POCO networking classes'
                ],
                'imports': 'Use #include "Poco/..." headers',
                'best_practices': 'Leverage POCO\'s cross-platform capabilities, use proper logging, implement configuration management'
            },
            
            # C# Frameworks
            'aspnetcore': {
                'conventions': 'ASP.NET Core MVC/API style',
                'patterns': [
                    'Use dependency injection container',
                    'Implement middleware pipeline properly',
                    'Follow RESTful API conventions',
                    'Use action filters for cross-cutting concerns'
                ],
                'imports': 'Use Microsoft.AspNetCore namespaces',
                'best_practices': 'Configure services properly in Startup, use proper async/await, implement proper authorization'
            },
            'entityframework': {
                'conventions': 'Entity Framework Core ORM style',
                'patterns': [
                    'Use Code First migrations',
                    'Implement DbContext properly',
                    'Follow repository pattern',
                    'Use LINQ for queries'
                ],
                'imports': 'Use Microsoft.EntityFrameworkCore namespaces',
                'best_practices': 'Avoid tracking for read-only queries, use proper loading strategies, implement unit of work pattern'
            },
            'wpf': {
                'conventions': 'WPF MVVM pattern style',
                'patterns': [
                    'Implement MVVM pattern properly',
                    'Use data binding effectively',
                    'Follow command pattern for actions',
                    'Use dependency properties'
                ],
                'imports': 'Use System.Windows namespaces',
                'best_practices': 'Avoid code-behind logic, use proper INotifyPropertyChanged, leverage data templates'
            },
            'xamarin': {
                'conventions': 'Xamarin cross-platform mobile style',
                'patterns': [
                    'Use Xamarin.Forms for UI',
                    'Implement platform-specific code properly',
                    'Follow MVVM pattern',
                    'Use dependency service for platform features'
                ],
                'imports': 'Use Xamarin.Forms namespaces',
                'best_practices': 'Optimize for mobile performance, handle platform differences, use proper navigation patterns'
            },
            
            # Go Frameworks
            'gin': {
                'conventions': 'Gin web framework style',
                'patterns': [
                    'Use gin.Context for request handling',
                    'Implement middleware with gin.HandlerFunc',
                    'Follow RESTful routing patterns',
                    'Use binding for request validation'
                ],
                'imports': 'Use import "github.com/gin-gonic/gin"',
                'best_practices': 'Use gin mode appropriately, implement proper error handling, leverage built-in middleware'
            },
            'echo': {
                'conventions': 'Echo web framework style',
                'patterns': [
                    'Use echo.Context for handling',
                    'Implement middleware properly',
                    'Follow routing group patterns',
                    'Use binding and validation'
                ],
                'imports': 'Use import "github.com/labstack/echo/v4"',
                'best_practices': 'Configure middleware order properly, use proper HTTP error handling, implement request ID tracking'
            },
            'fiber': {
                'conventions': 'Fiber Express-like style',
                'patterns': [
                    'Use fiber.Ctx for context',
                    'Implement middleware chains',
                    'Follow Express-like routing',
                    'Use built-in features effectively'
                ],
                'imports': 'Use import "github.com/gofiber/fiber/v2"',
                'best_practices': 'Leverage Fiber\'s performance, use proper error handling, implement rate limiting'
            },
            'beego': {
                'conventions': 'Beego MVC framework style',
                'patterns': [
                    'Use Controller structs',
                    'Implement routers properly',
                    'Follow MVC architecture',
                    'Use ORM for database'
                ],
                'imports': 'Use import "github.com/beego/beego/v2"',
                'best_practices': 'Use built-in cache, implement proper validation, leverage auto-routing'
            },
            'revel': {
                'conventions': 'Revel full-stack framework style',
                'patterns': [
                    'Use Controller embedding',
                    'Implement interceptors',
                    'Follow convention-based routing',
                    'Use Revel templates'
                ],
                'imports': 'Use import "github.com/revel/revel"',
                'best_practices': 'Leverage hot reload in development, use proper validation, implement filters effectively'
            },
            
            # Rust Frameworks
            'actix': {
                'conventions': 'Actix actor-based web style',
                'patterns': [
                    'Use async/await patterns',
                    'Implement handlers with proper extractors',
                    'Follow actor model when needed',
                    'Use middleware properly'
                ],
                'imports': 'Use use actix_web::{web, App, HttpServer}',
                'best_practices': 'Leverage Rust\'s type system, use proper error handling with Result, implement efficient routing'
            },
            'rocket': {
                'conventions': 'Rocket type-safe web style',
                'patterns': [
                    'Use attribute macros for routes',
                    'Implement request guards',
                    'Follow type-safe patterns',
                    'Use Rocket\'s testing features'
                ],
                'imports': 'Use #[macro_use] extern crate rocket',
                'best_practices': 'Leverage compile-time guarantees, use proper error catchers, implement custom responders'
            },
            'tokio': {
                'conventions': 'Tokio async runtime style',
                'patterns': [
                    'Use async/await properly',
                    'Implement futures correctly',
                    'Handle spawning tasks',
                    'Use channels for communication'
                ],
                'imports': 'Use use tokio::{spawn, select, time}',
                'best_practices': 'Avoid blocking operations, use proper cancellation, implement structured concurrency'
            },
            'diesel': {
                'conventions': 'Diesel ORM compile-time safe style',
                'patterns': [
                    'Use schema macros',
                    'Implement migrations properly',
                    'Follow type-safe query building',
                    'Use connection pooling'
                ],
                'imports': 'Use use diesel::prelude::*',
                'best_practices': 'Leverage compile-time query checking, use proper transaction handling, implement efficient queries'
            },
            'serde': {
                'conventions': 'Serde serialization style',
                'patterns': [
                    'Use derive macros for structs',
                    'Implement custom serialization when needed',
                    'Handle errors properly',
                    'Use appropriate data formats'
                ],
                'imports': 'Use use serde::{Serialize, Deserialize}',
                'best_practices': 'Choose efficient formats, handle optional fields properly, use serde attributes effectively'
            },
            
            # Ruby Frameworks
            'rails': {
                'conventions': 'Rails convention-over-configuration style',
                'patterns': [
                    'Follow MVC architecture strictly',
                    'Use Rails generators appropriately',
                    'Implement RESTful resources',
                    'Follow Rails naming conventions'
                ],
                'imports': 'Rails autoloads most classes',
                'best_practices': 'Don\'t fight the framework, use ActiveRecord efficiently, implement proper concerns'
            },
            'sinatra': {
                'conventions': 'Sinatra minimalist DSL style',
                'patterns': [
                    'Use route blocks effectively',
                    'Keep it simple and lightweight',
                    'Implement helpers for reusable code',
                    'Use appropriate middleware'
                ],
                'imports': 'Use require "sinatra" or modular style',
                'best_practices': 'Keep routes focused, use proper HTTP verbs, implement error handling'
            },
            'hanami': {
                'conventions': 'Hanami clean architecture style',
                'patterns': [
                    'Follow clean architecture principles',
                    'Use actions for HTTP endpoints',
                    'Implement entities and repositories',
                    'Keep business logic separate'
                ],
                'imports': 'Use require "hanami"',
                'best_practices': 'Maintain clear boundaries, use interactors for business logic, test each layer independently'
            },
            'grape': {
                'conventions': 'Grape REST API DSL style',
                'patterns': [
                    'Use API versioning properly',
                    'Implement parameter validation',
                    'Follow RESTful conventions',
                    'Use entities for responses'
                ],
                'imports': 'Use require "grape"',
                'best_practices': 'Document APIs with Swagger, use proper error handling, implement authentication'
            },
            'roda': {
                'conventions': 'Roda routing tree style',
                'patterns': [
                    'Use routing tree effectively',
                    'Implement plugins as needed',
                    'Keep routes organized',
                    'Use Roda\'s performance features'
                ],
                'imports': 'Use require "roda"',
                'best_practices': 'Leverage routing tree performance, use plugins wisely, keep it simple'
            },
            
            # PHP Frameworks
            'laravel': {
                'conventions': 'Laravel elegant syntax style',
                'patterns': [
                    'Use Eloquent ORM properly',
                    'Implement service providers',
                    'Follow Laravel naming conventions',
                    'Use artisan commands'
                ],
                'imports': 'Use namespace App\\... and use statements',
                'best_practices': 'Leverage Laravel\'s features, use proper validation, implement queues for heavy tasks'
            },
            'symfony': {
                'conventions': 'Symfony component-based style',
                'patterns': [
                    'Use dependency injection',
                    'Implement services properly',
                    'Follow Symfony best practices',
                    'Use annotations or attributes'
                ],
                'imports': 'Use Symfony\\Component\\... namespaces',
                'best_practices': 'Configure services properly, use event system, implement proper security'
            },
            'codeigniter': {
                'conventions': 'CodeIgniter MVC style',
                'patterns': [
                    'Follow MVC pattern',
                    'Use CI loader properly',
                    'Implement models and controllers',
                    'Use helpers and libraries'
                ],
                'imports': 'CI autoloads based on configuration',
                'best_practices': 'Keep controllers thin, use models for data logic, implement proper validation'
            },
            'slim': {
                'conventions': 'Slim microframework style',
                'patterns': [
                    'Use PSR-7 request/response',
                    'Implement middleware properly',
                    'Follow PSR standards',
                    'Keep it lightweight'
                ],
                'imports': 'Use Slim\\... namespaces',
                'best_practices': 'Use dependency container, implement proper error handling, follow PSR standards'
            },
            'yii': {
                'conventions': 'Yii component-based style',
                'patterns': [
                    'Use ActiveRecord properly',
                    'Implement components and modules',
                    'Follow Yii conventions',
                    'Use behaviors and events'
                ],
                'imports': 'Use yii\\... namespaces',
                'best_practices': 'Configure components properly, use caching effectively, implement RBAC for authorization'
            },
            
            # Swift Frameworks
            'vapor': {
                'conventions': 'Vapor server-side Swift style',
                'patterns': [
                    'Use async/await patterns',
                    'Implement routes with proper handlers',
                    'Use Fluent ORM for database',
                    'Follow Swift conventions'
                ],
                'imports': 'Use import Vapor',
                'best_practices': 'Leverage Swift\'s type safety, use proper error handling, implement middleware effectively'
            },
            'perfect': {
                'conventions': 'Perfect server framework style',
                'patterns': [
                    'Use Perfect HTTP server',
                    'Implement routes properly',
                    'Handle requests and responses',
                    'Use Perfect libraries'
                ],
                'imports': 'Use import PerfectHTTP',
                'best_practices': 'Handle threading properly, use Perfect\'s utilities, implement proper error handling'
            },
            'kitura': {
                'conventions': 'Kitura IBM Swift style',
                'patterns': [
                    'Use Router for routing',
                    'Implement middleware',
                    'Follow Kitura patterns',
                    'Use Codable for JSON'
                ],
                'imports': 'Use import Kitura',
                'best_practices': 'Use Swift package manager, implement proper logging, leverage Swift features'
            },
            'swiftui': {
                'conventions': 'SwiftUI declarative UI style',
                'patterns': [
                    'Use declarative syntax',
                    'Implement proper state management',
                    'Follow SwiftUI view composition',
                    'Use property wrappers'
                ],
                'imports': 'Use import SwiftUI',
                'best_practices': 'Keep views simple, use proper state management, avoid massive views'
            },
            
            # Kotlin Frameworks
            'ktor': {
                'conventions': 'Ktor coroutine-based style',
                'patterns': [
                    'Use coroutines for async operations',
                    'Implement features as plugins',
                    'Follow Ktor routing DSL',
                    'Use serialization properly'
                ],
                'imports': 'Use import io.ktor...',
                'best_practices': 'Leverage Kotlin coroutines, use proper error handling, implement testing with TestEngine'
            },
            'android': {
                'conventions': 'Android Jetpack modern style',
                'patterns': [
                    'Use ViewModel and LiveData',
                    'Implement proper lifecycle handling',
                    'Follow MVVM architecture',
                    'Use Jetpack components'
                ],
                'imports': 'Use androidx.* imports',
                'best_practices': 'Handle configuration changes, avoid memory leaks, use proper navigation'
            },
            'spring-kotlin': {
                'conventions': 'Spring Boot with Kotlin style',
                'patterns': [
                    'Use Kotlin DSL for configuration',
                    'Leverage null safety',
                    'Use data classes for DTOs',
                    'Implement coroutines for async'
                ],
                'imports': 'Use org.springframework imports',
                'best_practices': 'Use Kotlin idioms with Spring, leverage extension functions, implement proper null handling'
            },
            
            # React Ecosystem
            'redux': {
                'conventions': 'Redux predictable state container',
                'patterns': [
                    'Follow flux architecture pattern',
                    'Keep reducers pure functions',
                    'Use action creators for consistency',
                    'Implement Redux Toolkit for modern Redux'
                ],
                'imports': 'Use import { createStore, combineReducers } from "redux"',
                'best_practices': 'Normalize state shape, use Redux DevTools, avoid mutations in reducers'
            },
            'mobx': {
                'conventions': 'MobX reactive state management',
                'patterns': [
                    'Use decorators or makeObservable',
                    'Keep components observer wrapped',
                    'Use computed values for derived state',
                    'Implement actions for state mutations'
                ],
                'imports': 'Use import { observable, action, computed } from "mobx"',
                'best_practices': 'Keep stores focused, use strict mode, avoid overusing observables'
            },
            'react-query': {
                'conventions': 'React Query server state management',
                'patterns': [
                    'Use queries for data fetching',
                    'Implement mutations for updates',
                    'Configure proper cache times',
                    'Use query invalidation effectively'
                ],
                'imports': 'Use import { useQuery, useMutation } from "react-query"',
                'best_practices': 'Leverage caching, handle loading/error states, use optimistic updates wisely'
            },
            'recoil': {
                'conventions': 'Recoil experimental state management',
                'patterns': [
                    'Use atoms for state units',
                    'Implement selectors for derived state',
                    'Follow atom family patterns',
                    'Use hooks for state access'
                ],
                'imports': 'Use import { atom, selector, useRecoilState } from "recoil"',
                'best_practices': 'Keep atoms small and focused, use selectors for computed values, handle async selectors properly'
            },
            
            # Vue Ecosystem
            'vuex': {
                'conventions': 'Vuex centralized state management',
                'patterns': [
                    'Use modules for organization',
                    'Follow mutation/action pattern',
                    'Keep mutations synchronous',
                    'Use getters for computed state'
                ],
                'imports': 'Use import { createStore } from "vuex"',
                'best_practices': 'Use namespaced modules, follow strict mode in development, avoid direct state mutations'
            },
            'pinia': {
                'conventions': 'Pinia intuitive state management',
                'patterns': [
                    'Use composition API style',
                    'Define stores with defineStore',
                    'Use getters for computed values',
                    'Implement actions for logic'
                ],
                'imports': 'Use import { defineStore } from "pinia"',
                'best_practices': 'Keep stores modular, use TypeScript for type safety, leverage Vue devtools integration'
            },
            'nuxt': {
                'conventions': 'Nuxt.js full-stack Vue framework',
                'patterns': [
                    'Follow file-based routing',
                    'Use asyncData for SSR data',
                    'Implement middleware properly',
                    'Use modules for functionality'
                ],
                'imports': 'Auto-imports are configured by Nuxt',
                'best_practices': 'Optimize for SSR/SSG, use proper head management, implement error pages'
            },
            
            # Angular Ecosystem
            'ngrx': {
                'conventions': 'NgRx reactive state management',
                'patterns': [
                    'Follow Redux pattern strictly',
                    'Use effects for side effects',
                    'Implement selectors for queries',
                    'Use entity adapter for collections'
                ],
                'imports': 'Use import { Store, createAction, createReducer } from "@ngrx/store"',
                'best_practices': 'Use facade pattern for complex features, leverage DevTools, implement proper error handling in effects'
            },
            'rxjs': {
                'conventions': 'RxJS reactive extensions',
                'patterns': [
                    'Use operators for transformations',
                    'Follow observable patterns',
                    'Implement proper subscription management',
                    'Use subjects appropriately'
                ],
                'imports': 'Use import { Observable, Subject, of, from } from "rxjs"',
                'best_practices': 'Unsubscribe to prevent memory leaks, use async pipe in templates, avoid nested subscribes'
            },
            
            # Zig Frameworks
            'zap': {
                'conventions': 'Zap web server style',
                'patterns': [
                    'Use const for server and router instances',
                    'Follow Zig style for error handling with try',
                    'Use explicit allocator management',
                    'Use idiomatic route handlers with context parameter'
                ],
                'imports': 'Use @import("zap") for imports',
                'best_practices': 'Follow memory safety practices with allocators'
            },
            'ziggy': {
                'conventions': 'Ziggy HTTP framework style',
                'patterns': [
                    'Use const for router definitions',
                    'Handle errors with try/catch or return',
                    'Use proper response builders',
                    'Explicit memory management'
                ],
                'imports': 'Use @import("ziggy") for imports',
                'best_practices': 'Use consistent error handling with try/catch'
            },
            
            # Nim Frameworks
            'jester': {
                'conventions': 'Jester web framework style',
                'patterns': [
                    'Use routes: block for route definitions',
                    'Return responses with resp function',
                    'Use @"" for routes with params',
                    'Async compatibility with {.async.} pragma'
                ],
                'imports': 'Use import jester for imports',
                'best_practices': 'Handle exceptions properly with try/except'
            },
            'karax': {
                'conventions': 'Karax UI framework style',
                'patterns': [
                    'Use buildHtml: for DOM construction',
                    'Follow reactive style for state management',
                    'Use proper event handlers',
                    'Define components as procs returning VNode'
                ],
                'imports': 'Use import karax/[karaxdsl, vdom]',
                'best_practices': 'Minimize DOM manipulations for performance'
            },
            
            # Crystal Frameworks
            'lucky': {
                'conventions': 'Lucky web framework style',
                'patterns': [
                    'Use class inheritance from Lucky::Action',
                    'Define routes in src/actions',
                    'Use method_name.cr naming pattern',
                    'Follow Ruby-like syntax conventions'
                ],
                'imports': 'Use require "lucky"',
                'best_practices': 'Use type annotations for method parameters'
            },
            'kemal': {
                'conventions': 'Kemal web framework style',
                'patterns': [
                    'Define routes with get, post, put, etc.',
                    'Use env.response for HTTP responses',
                    'Use env.params for request parameters',
                    'Handle exceptions with error block'
                ],
                'imports': 'Use require "kemal"',
                'best_practices': 'Set content_type for responses'
            },
            
            # Haskell Frameworks
            'yesod': {
                'conventions': 'Yesod web framework style',
                'patterns': [
                    'Use mkYesod and instance Yesod',
                    'Define routes with pattern syntax',
                    'Use hamlet templates with #{var}',
                    'Handle forms with renderForm'
                ],
                'imports': 'Use import Yesod',
                'best_practices': 'Use type-safe URLs with proper routing'
            },
            'servant': {
                'conventions': 'Servant API framework style',
                'patterns': [
                    'Define API types with type API =',
                    'Use type-level operators like :>',
                    'Implement handlers with proper types',
                    'Use proper content-type combinators'
                ],
                'imports': 'Use import Servant.API, Servant.Server',
                'best_practices': 'Leverage type system for API guarantees'
            },
            
            # F# Frameworks
            'giraffe': {
                'conventions': 'Giraffe web framework style',
                'patterns': [
                    'Compose HttpHandlers with >=>',
                    'Use route and routef for routing',
                    'Use choose for alternative routes',
                    'JSON serialization with json'
                ],
                'imports': 'Use open Giraffe',
                'best_practices': 'Use computation expressions for complex flows'
            },
            'saturn': {
                'conventions': 'Saturn web framework style',
                'patterns': [
                    'Use application CE for app config',
                    'Use controller CE for controllers',
                    'Define routes with router CE',
                    'Configure endpoints with scope'
                ],
                'imports': 'Use open Saturn',
                'best_practices': 'Group related endpoints with scope'
            },
            
            # Erlang Frameworks
            'cowboy': {
                'conventions': 'Cowboy web server style',
                'patterns': [
                    'Define routes as tuples',
                    'Implement handlers with init/terminate',
                    'Pattern match on requests',
                    'Use proper response tuples'
                ],
                'imports': 'Use -include_lib("cowboy/include/cowboy.hrl").',
                'best_practices': 'Use proper OTP principles for state management'
            },
            'otp': {
                'conventions': 'OTP behavior style',
                'patterns': [
                    'Use -behaviour(gen_server).',
                    'Implement callback functions',
                    'Use proper state management',
                    'Pattern match for message handling'
                ],
                'imports': 'Use proper OTP includes',
                'best_practices': 'Follow OTP principles for fault tolerance'
            },
            
            # R Frameworks
            'shiny': {
                'conventions': 'Shiny web app style',
                'patterns': [
                    'Split UI and server logic',
                    'Use reactive expressions',
                    'Use input$ for inputs',
                    'Use output$ for outputs'
                ],
                'imports': 'Use library(shiny)',
                'best_practices': 'Minimize code in reactive contexts'
            },
            'tidyverse': {
                'conventions': 'Tidyverse data analysis style',
                'patterns': [
                    'Use pipe operator %>%',
                    'Use tibbles instead of data frames',
                    'Use ggplot2 with proper aesthetics',
                    'Use tidyr functions for reshaping'
                ],
                'imports': 'Use library(tidyverse)',
                'best_practices': 'Chain operations with the pipe operator'
            },
            
            # Julia Frameworks
            'genie': {
                'conventions': 'Genie web framework style',
                'patterns': [
                    'Define routes with route()',
                    'Use decorators like @get, @post',
                    'Use params for request parameters',
                    'Return response with html()'
                ],
                'imports': 'Use using Genie',
                'best_practices': 'Use MVC architecture with models/controllers/views'
            },
            'flux': {
                'conventions': 'Flux ML framework style',
                'patterns': [
                    'Define models with Chain()',
                    'Use Dense, Conv layers',
                    'Use Flux.train! for training',
                    'Use CUDA for GPU acceleration'
                ],
                'imports': 'Use using Flux',
                'best_practices': 'Use batches for better performance'
            },
            
            # Terraform Providers
            'terraform_aws': {
                'conventions': 'Terraform AWS provider style',
                'patterns': [
                    'Define provider before resources',
                    'Use consistent naming convention',
                    'Use variables and locals',
                    'Reference resources with proper syntax'
                ],
                'imports': 'Use provider "aws" block',
                'best_practices': 'Use modules for reusable components'
            },
            'terraform_azure': {
                'conventions': 'Terraform Azure provider style',
                'patterns': [
                    'Define azurerm provider',
                    'Create resource groups first',
                    'Use proper naming convention',
                    'Reference resources with proper syntax'
                ],
                'imports': 'Use provider "azurerm" block',
                'best_practices': 'Use tags for resource organization'
            },
            
            # Ansible Modules
            'ansible_apt': {
                'conventions': 'Ansible Apt module style',
                'patterns': [
                    'Use yaml indentation consistently',
                    'Use name parameter for packages',
                    'Use state: present/absent/latest',
                    'Use become: true for sudo'
                ],
                'imports': 'N/A - Use proper module reference',
                'best_practices': 'Use update_cache when installing packages'
            },
            'ansible_service': {
                'conventions': 'Ansible Service module style',
                'patterns': [
                    'Use name parameter for service name',
                    'Use state: started/stopped/restarted',
                    'Use enabled: yes/no for boot behavior',
                    'Use become: true for sudo'
                ],
                'imports': 'N/A - Use proper module reference',
                'best_practices': 'Ensure dependent packages are installed first'
            }
        }
        
        return framework_guidance.get(framework_name, {
            'conventions': 'Follow framework conventions',
            'patterns': ['Use appropriate patterns for the framework'],
            'imports': 'Follow import conventions for the framework',
            'best_practices': 'Follow best practices for the framework'
        })
    
    def _get_language_guidance(self, language: LanguageType) -> Dict[str, Any]:
        """
        Get language-specific guidance for LLM patch generation.
        
        Args:
            language: Programming language
            
        Returns:
            Guidance dictionary
        """
        guidance_map = {
            LanguageType.PYTHON: {
                'style_guide': 'PEP 8',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use PascalCase for classes',
                    'Prefer list comprehensions when appropriate',
                    'Use context managers (with statements) for resource management'
                ],
                'error_handling': 'Use try/except blocks with specific exception types',
                'imports': 'Place imports at top, group standard library, third-party, local'
            },
            LanguageType.JAVASCRIPT: {
                'style_guide': 'Airbnb or Standard',
                'common_patterns': [
                    'Use camelCase for variables and functions',
                    'Use PascalCase for constructors and classes',
                    'Prefer const/let over var',
                    'Use arrow functions for callbacks'
                ],
                'error_handling': 'Use try/catch blocks or Promise .catch()',
                'imports': 'Use ES6 import/export syntax'
            },
            LanguageType.JAVA: {
                'style_guide': 'Google Java Style Guide',
                'common_patterns': [
                    'Use camelCase for variables and methods',
                    'Use PascalCase for classes',
                    'Use ALL_CAPS for constants',
                    'Follow bean naming conventions'
                ],
                'error_handling': 'Use try/catch blocks with specific exception types',
                'imports': 'Organize imports and avoid wildcards'
            },
            LanguageType.GO: {
                'style_guide': 'Go official style guide',
                'common_patterns': [
                    'Use camelCase for exported functions',
                    'Use lowercase for package-private',
                    'Follow receiver naming conventions',
                    'Use short variable names in small scopes'
                ],
                'error_handling': 'Check errors explicitly, return error as last value',
                'imports': 'Group standard library, third-party, local'
            },
            LanguageType.RUST: {
                'style_guide': 'Rust official style guide',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use PascalCase for types and traits',
                    'Use SCREAMING_SNAKE_CASE for constants',
                    'Prefer ? operator for error propagation'
                ],
                'error_handling': 'Use Result<T, E> and Option<T> types',
                'imports': 'Use explicit use statements'
            },
            LanguageType.ZIG: {
                'style_guide': 'Zig style guide',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use PascalCase for types',
                    'Explicit error handling with error unions',
                    'Comptime for compile-time computation'
                ],
                'error_handling': 'Use error unions and try/catch',
                'imports': 'Use @import() for modules'
            },
            LanguageType.NIM: {
                'style_guide': 'Nim style guide',
                'common_patterns': [
                    'Use camelCase for procedures and variables',
                    'Use PascalCase for types',
                    'Prefer result types for error handling',
                    'Use pragmas for compiler hints'
                ],
                'error_handling': 'Use exceptions or Option/Result types',
                'imports': 'Use import statements'
            },
            LanguageType.CRYSTAL: {
                'style_guide': 'Crystal style guide (Ruby-like)',
                'common_patterns': [
                    'Use snake_case for methods and variables',
                    'Use PascalCase for classes and modules',
                    'Use SCREAMING_SNAKE_CASE for constants',
                    'Type inference with optional type annotations'
                ],
                'error_handling': 'Use exceptions with begin/rescue/end',
                'imports': 'Use require statements'
            },
            LanguageType.HASKELL: {
                'style_guide': 'Haskell style guide',
                'common_patterns': [
                    'Use camelCase for functions and variables',
                    'Use PascalCase for types and constructors',
                    'Pattern matching for control flow',
                    'Pure functions by default'
                ],
                'error_handling': 'Use Maybe, Either, or custom monads',
                'imports': 'Use import statements with qualified names'
            },
            LanguageType.FSHARP: {
                'style_guide': 'F# style guide',
                'common_patterns': [
                    'Use camelCase for values and functions',
                    'Use PascalCase for types and modules',
                    'Prefer immutability',
                    'Use pattern matching extensively'
                ],
                'error_handling': 'Use Result<\'T,\'TError> or Option types',
                'imports': 'Use open statements'
            },
            LanguageType.ERLANG: {
                'style_guide': 'Erlang style guide',
                'common_patterns': [
                    'Use snake_case for functions and atoms',
                    'Use PascalCase for variables',
                    'Actor model with message passing',
                    'Pattern matching in function heads'
                ],
                'error_handling': 'Use pattern matching on {ok, Result} or {error, Reason}',
                'imports': 'Use -include and -import directives'
            },
            LanguageType.SQL: {
                'style_guide': 'SQL style conventions',
                'common_patterns': [
                    'Use UPPERCASE for SQL keywords',
                    'Use snake_case for table and column names',
                    'Proper indentation for readability',
                    'Use meaningful aliases'
                ],
                'error_handling': 'Handle NULL values and use transactions',
                'imports': 'N/A - Use proper schema references'
            },
            LanguageType.BASH: {
                'style_guide': 'Bash style guide (Google)',
                'common_patterns': [
                    'Use lowercase with underscores for variables',
                    'Use uppercase for environment variables',
                    'Quote variables to prevent word splitting',
                    'Use [[ ]] for conditionals'
                ],
                'error_handling': 'Check exit codes, use set -e, trap errors',
                'imports': 'Use source or . to include scripts'
            },
            LanguageType.POWERSHELL: {
                'style_guide': 'PowerShell style guide',
                'common_patterns': [
                    'Use PascalCase for cmdlets (Verb-Noun)',
                    'Use camelCase for variables',
                    'Use approved verbs for functions',
                    'Explicit type declarations when needed'
                ],
                'error_handling': 'Use try/catch blocks and -ErrorAction',
                'imports': 'Use Import-Module'
            },
            LanguageType.LUA: {
                'style_guide': 'Lua style guide',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use PascalCase for classes/modules',
                    'Local variables preferred over global',
                    'Tables for data structures'
                ],
                'error_handling': 'Use pcall/xpcall for protected calls',
                'imports': 'Use require() for modules'
            },
            LanguageType.R: {
                'style_guide': 'R style guide (tidyverse)',
                'common_patterns': [
                    'Use snake_case for variables and functions',
                    'Use <- for assignment',
                    'Vectorized operations preferred',
                    'Use explicit returns'
                ],
                'error_handling': 'Use tryCatch() blocks',
                'imports': 'Use library() or require()'
            },
            LanguageType.MATLAB: {
                'style_guide': 'MATLAB style guide',
                'common_patterns': [
                    'Use camelCase for variables and functions',
                    'Use uppercase for constants',
                    'Vectorize operations when possible',
                    'Preallocate arrays'
                ],
                'error_handling': 'Use try/catch blocks',
                'imports': 'Use addpath or import'
            },
            LanguageType.JULIA: {
                'style_guide': 'Julia style guide',
                'common_patterns': [
                    'Use snake_case for functions and variables',
                    'Use PascalCase for types and modules',
                    'Type annotations for performance',
                    'Multiple dispatch for polymorphism'
                ],
                'error_handling': 'Use try/catch blocks or @error macro',
                'imports': 'Use using or import'
            },
            LanguageType.TERRAFORM: {
                'style_guide': 'Terraform style conventions',
                'common_patterns': [
                    'Use snake_case for all names',
                    'Group related resources',
                    'Use meaningful resource names',
                    'Pin provider versions'
                ],
                'error_handling': 'Use validation blocks and preconditions',
                'imports': 'Use module blocks'
            },
            LanguageType.ANSIBLE: {
                'style_guide': 'Ansible best practices',
                'common_patterns': [
                    'Use snake_case for variables',
                    'Prefix role variables with role name',
                    'Use meaningful task names',
                    'YAML formatting with proper indentation'
                ],
                'error_handling': 'Use failed_when, ignore_errors, and block/rescue',
                'imports': 'Use include_tasks or import_playbook'
            },
            LanguageType.YAML: {
                'style_guide': 'YAML style guide',
                'common_patterns': [
                    'Consistent indentation (2 or 4 spaces)',
                    'Use hyphens for lists',
                    'Quote strings when necessary',
                    'Avoid tabs'
                ],
                'error_handling': 'Validate schema compliance',
                'imports': 'Use anchors and aliases for reuse'
            },
            LanguageType.JSON: {
                'style_guide': 'JSON formatting conventions',
                'common_patterns': [
                    'Use double quotes for strings',
                    'No trailing commas',
                    'Consistent indentation',
                    'Valid data types only'
                ],
                'error_handling': 'Ensure valid JSON syntax',
                'imports': 'N/A - Use references or includes at application level'
            },
            LanguageType.DOCKERFILE: {
                'style_guide': 'Dockerfile best practices',
                'common_patterns': [
                    'Use UPPERCASE for instructions',
                    'One instruction per line',
                    'Minimize layers',
                    'Use specific base image tags'
                ],
                'error_handling': 'Use HEALTHCHECK and proper error codes',
                'imports': 'Use FROM for base images, COPY for files'
            }
        }
        
        return guidance_map.get(language, {
            'style_guide': 'Follow language conventions',
            'common_patterns': [],
            'error_handling': 'Use appropriate error handling for the language',
            'imports': 'Organize imports properly'
        })


def create_multi_language_detector() -> MultiLanguageFrameworkDetector:
    """Factory function to create a multi-language framework detector."""
    return MultiLanguageFrameworkDetector()


if __name__ == "__main__":
    # Test the detector
    print("Testing Multi-Language Framework Detector")
    print("=========================================")
    
    detector = create_multi_language_detector()
    
    # Test Python code
    python_code = '''
import os
from django.db import models
from django.contrib.auth.models import User

class BlogPost(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
'''
    
    result = detector.detect_language_and_frameworks(
        file_path="blog/models.py",
        source_code=python_code
    )
    
    print(f"Language: {result.language.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Frameworks: {[f.name for f in result.frameworks]}")
    print(f"Features: {result.language_features}")
    
    # Test JavaScript code
    js_code = '''
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BlogList = () => {
    const [posts, setPosts] = useState([]);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        const fetchPosts = async () => {
            try {
                const response = await axios.get('/api/posts');
                setPosts(response.data);
            } catch (error) {
                console.error('Error fetching posts:', error);
            } finally {
                setLoading(false);
            }
        };
        
        fetchPosts();
    }, []);
    
    return (
        <div className="blog-list">
            {loading ? (
                <p>Loading...</p>
            ) : (
                posts.map(post => (
                    <div key={post.id} className="blog-post">
                        <h2>{post.title}</h2>
                        <p>{post.content}</p>
                    </div>
                ))
            )}
        </div>
    );
};

export default BlogList;
'''
    
    result2 = detector.detect_language_and_frameworks(
        file_path="src/components/BlogList.jsx",
        source_code=js_code
    )
    
    print(f"\nSecond test:")
    print(f"Language: {result2.language.value}")
    print(f"Confidence: {result2.confidence:.2f}")
    print(f"Frameworks: {[f.name for f in result2.frameworks]}")
    
    # Test LLM context generation
    context = detector.get_llm_context_for_language(result)
    print(f"\nLLM Context: {json.dumps(context, indent=2)}")