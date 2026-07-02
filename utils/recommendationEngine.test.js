const { getRecommendation } = require('./recommendationEngine');

const mockDatabase = [
  { disease_id: 'maize_rust', treatments: [{ name: 'Amistar Xtra' }] },
  { disease_id: 'maize_leaf_blight', treatments: [{ name: 'Dithane M-45' }] }
];

console.log("Running Recommendation Engine Tests...\n");

// Test 1: Low Confidence (< 0.75)
let res1 = getRecommendation({ disease: 'maize_rust', confidence: 0.74 }, mockDatabase);
console.assert(res1 === "uncertain", "Test 1 Failed: Should return 'uncertain' for low confidence");

// Test 2: Exact lookup success (>= 0.75)
let res2 = getRecommendation({ disease: 'maize_rust', confidence: 0.85 }, mockDatabase);
console.assert(res2[0].name === 'Amistar Xtra', "Test 2 Failed: Should return exact treatment");

// Test 3: Missing disease
try {
  getRecommendation({ disease: 'banana_disease', confidence: 0.99 }, mockDatabase);
  console.error("Test 3 Failed: Should have thrown an error");
} catch (error) {
  console.assert(error.message === "Disease 'banana_disease' not found in verified database", "Test 3 Failed: Wrong error message");
}

// Test 4: Missing confidence field
let res4 = getRecommendation({ disease: 'maize_rust' }, mockDatabase);
console.assert(res4 === "uncertain", "Test 4 Failed: Should return 'uncertain' for missing confidence");

console.log("All tests passed! Deterministic strict rules enforced.");
