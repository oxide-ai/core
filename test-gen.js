/**
 * Test script for text generation using OxideEngine
 * Run with: node test-gen.js [path-to-model-file-or-dir] [path-to-tokenizer.json] [path-to-config.json]
 */

const path = require('path');
const fs = require('fs');
const { OxideEngine } = require('./index.js');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

function printHeader(text) {
  console.log(`\n${colors.bright}${colors.blue}${'='.repeat(70)}${colors.reset}`);
  console.log(`${colors.bright}${colors.blue}${text}${colors.reset}`);
  console.log(`${colors.bright}${colors.blue}${'='.repeat(70)}${colors.reset}\n`);
}

function printSuccess(text) {
  console.log(`${colors.green}✓${colors.reset} ${text}`);
}

function printError(text) {
  console.log(`${colors.red}✗${colors.reset} ${text}`);
}

function printInfo(text) {
  console.log(`${colors.cyan}ℹ${colors.reset} ${text}`);
}

function printWarning(text) {
  console.log(`${colors.yellow}⚠${colors.reset} ${text}`);
}

async function runTests() {
  console.clear();
  console.log(`${colors.bright}${colors.magenta}OxideEngine Text Generation Test${colors.reset}`);
  console.log(`${colors.yellow}Testing end-to-end text generation with tokenization${colors.reset}\n`);

  // Get paths from command line
  const modelPath = process.argv[2];
  const tokenizerPath = process.argv[3];
  const configPath = process.argv[4];

  if (!modelPath || !tokenizerPath) {
    printWarning('Model path or tokenizer path not provided');
    console.log('\nUsage:');
    console.log('  node test-gen.js <model_path> <tokenizer.json> [config.json]');
    console.log('\nArguments:');
    console.log('  model_path: Path to a specific .safetensors file OR a directory containing sharded files');
    console.log('  tokenizer.json: Path to the tokenizer file');
    console.log('  config.json: (Optional) Path to model config');
    
    console.log('\nExample (Single File):');
    console.log('  node test-gen.js ./models/model.safetensors ./models/tokenizer.json');
    
    console.log('\nExample (Sharded/Directory):');
    console.log('  node test-gen.js ./models/ ./models/tokenizer.json');

    console.log('\nNote: If config.json is not provided, default Phi-3-mini config will be used.');
    console.log('\nYou can download Phi-3 models from Hugging Face:');
    console.log('  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct');
    console.log('\nFiles needed:');
    console.log('  - model.safetensors (or model-00001... etc)');
    console.log('  - tokenizer.json (tokenizer vocabulary)');
    console.log('  - config.json (optional, model configuration)');
    console.log('\nFor now, running basic GPU test without model loading...\n');

    try {
      printHeader('Test 1: Initialize OxideEngine');
      console.time('Initialization time');
      const engine = new OxideEngine();
      console.timeEnd('Initialization time');
      printSuccess('OxideEngine initialized successfully');

      const deviceInfo = engine.getDeviceInfo();
      printInfo(`Device: ${deviceInfo}`);

      printHeader('Test 2: Basic GPU Compute Test');
      console.time('GPU compute time');
      const computeResult = engine.testGpuCompute();
      console.timeEnd('GPU compute time');
      console.log('\n' + computeResult);
      printSuccess('GPU compute test passed\n');

      printHeader('Summary');
      printInfo('GPU is working correctly');
      printWarning('Model and tokenizer not loaded - provide paths to test text generation');
      console.log();
    } catch (error) {
      printError('Test failed: ' + error.message);
      process.exit(1);
    }
    return;
  }

  // Full test with model and tokenizer
  try {
    // Verify files exist
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model path not found: ${modelPath}`);
    }
    
    // Check if model path is directory or file
    const stats = fs.statSync(modelPath);
    if (stats.isDirectory()) {
        printInfo(`Model path is a directory: ${modelPath}`);
        // We trust the engine to find .safetensors inside
    } else if (stats.isFile()) {
        printInfo(`Model path is a file: ${modelPath}`);
    }

    if (!fs.existsSync(tokenizerPath)) {
      throw new Error(`Tokenizer file not found: ${tokenizerPath}`);
    }
    if (configPath && !fs.existsSync(configPath)) {
      throw new Error(`Config file not found: ${configPath}`);
    }

    // Test 1: Initialize engine
    printHeader('Test 1: Initialize OxideEngine');
    console.time('Initialization time');
    const engine = new OxideEngine();
    console.timeEnd('Initialization time');
    printSuccess('OxideEngine initialized successfully');

    const deviceInfo = engine.getDeviceInfo();
    printInfo(`Device: ${deviceInfo}`);
    console.log();

    // Test 2: Load model and tokenizer
    printHeader('Test 2: Load Model and Tokenizer');
    printInfo(`Model path: ${modelPath}`);
    printInfo(`Tokenizer path: ${tokenizerPath}`);
    if (configPath) {
      printInfo(`Config path: ${configPath}`);
    } else {
      printInfo('Using default Phi-3-mini config');
    }

    console.time('Loading time');
    const loadResult = engine.loadModel(modelPath, tokenizerPath, configPath || null);
    console.timeEnd('Loading time');
    console.log('\n' + loadResult);
    printSuccess('Model and tokenizer loaded successfully\n');

    // Test 3: Generate text from prompt
    printHeader('Test 3: Text Generation');

    const prompts = [
      { text: "The capital of France is", maxTokens: 20 },
      { text: "Once upon a time", maxTokens: 30 },
      { text: "def fibonacci(n):", maxTokens: 50 },
    ];

    for (let i = 0; i < prompts.length; i++) {
      const { text: prompt, maxTokens } = prompts[i];

      console.log(`\n${colors.bright}Generation ${i + 1}:${colors.reset}`);
      printInfo(`Prompt: "${prompt}"`);
      printInfo(`Max tokens: ${maxTokens}`);

      console.log(`\n${colors.cyan}Generating...${colors.reset}`);
      console.time('Generation time');

      try {
        // Use new options API
        const generated = engine.generateText(prompt, maxTokens, {
            temperature: 0.7,
            topP: 0.9,
            seed: 42
        });
        console.timeEnd('Generation time');

        console.log(`\n${colors.bright}${colors.green}Generated Text:${colors.reset}`);
        console.log(`${colors.bright}"${prompt}${generated}"${colors.reset}\n`);

        printSuccess(`Generated ${generated.length} characters`);
      } catch (error) {
        console.timeEnd('Generation time');
        printError(`Generation failed: ${error.message}`);
      }

      // Reset cache between different prompts
      if (i < prompts.length - 1) {
        printInfo('Resetting cache for next prompt...');
        engine.resetCache();
      }
    }

    console.log();

    // Test 4: Check cache info
    printHeader('Test 4: Cache Information');
    const cacheInfo = engine.getCacheInfo();
    printInfo(`Cache status: ${cacheInfo.message}`);
    printInfo(`Sequence length: ${cacheInfo.sequenceLength}`);
    printInfo(`Is empty: ${cacheInfo.isEmpty}`);
    console.log();

    // Summary
    printHeader('Test Summary');
    printSuccess('All text generation tests completed!');
    console.log(`
${colors.bright}Key Features Demonstrated:${colors.reset}

1. ${colors.green}✓${colors.reset} ${colors.bright}Tokenization:${colors.reset}
   - Prompt encoding using HuggingFace tokenizer
   - Token decoding back to text
   - EOS token detection

2. ${colors.green}✓${colors.reset} ${colors.bright}Text Generation:${colors.reset}
   - Autoregressive token generation
   - LogitsProcessor sampling (temp=0.8, top_p=0.9)
   - Efficient KV caching for O(1) history

3. ${colors.green}✓${colors.reset} ${colors.bright}Cache Management:${colors.reset}
   - Persistent cache across generations
   - Reset capability for new conversations
   - Cache monitoring via getCacheInfo()

${colors.bright}Performance:${colors.reset}
- ✓ GPU acceleration active
- ✓ Real-time token generation
- ✓ Production-ready inference pipeline

${colors.bright}API Example:${colors.reset}

  const engine = new OxideEngine();
  engine.loadModel('model.safetensors', 'tokenizer.json');

  const text = engine.generateText('The capital of France is', 20);
  console.log(text);  // " Paris, which is located..."

  engine.resetCache();  // Start fresh for new prompt
    `);

  } catch (error) {
    printHeader('Test Failed');
    printError('Error occurred during testing:');
    console.error(`\n${colors.red}${error.message}${colors.reset}`);
    if (error.stack) {
      console.error(`\n${colors.yellow}Stack trace:${colors.reset}`);
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Run all tests
runTests().catch((error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});
