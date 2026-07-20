import js from "@eslint/js";

export default [
  js.configs.recommended,
  {
    ignores: [
      "node_modules/",
      "*.py",
      "skcq/**/*.py",
      "scripts/",
      "experiments/",
      "tests/",
      "poc/",
    ],
  },
  {
    files: ["skcq/vq/dashboard/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        document: "readonly",
        window: "readonly",
        console: "readonly",
        fetch: "readonly",
        setInterval: "readonly",
        Plotly: "readonly",
        $: "readonly",
        jQuery: "readonly",
      },
    },
    rules: {
      "max-len": ["warn", {
        code: 100,
        ignoreStrings: true,
        ignoreTemplateLiterals: true,
      }],
      "no-unused-vars": "off",
      "no-undef": "off",
      "no-redeclare": "off",
      "prefer-const": "error",
      "no-var": "error",
      eqeqeq: "warn",
    },
  },
];
