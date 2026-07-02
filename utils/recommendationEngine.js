/**
 * Strict Recommendation Engine
 * 
 * Rules:
 * - No guessing, no fuzzy matching.
 * - No generated advice.
 * - Deterministic and exact lookups only.
 *
 * @param {Object} input - Format: { "disease": "string", "confidence": float }
 * @param {Array} database - Verified local disease database array
 * @param {Array} products - Verified product database array
 * @returns {Array|string} Treatment array, "uncertain" string, or throws Error
 */
function getRecommendation(input, database, products) {
  // 1. Check confidence threshold
  if (typeof input.confidence !== 'number' || input.confidence < 0.75) {
    return "uncertain";
  }

  // 2. Perform exact lookup in local database
  // Strict matching only, no fallback to lowercase or regex
  const exactMatch = database.find(entry => entry.disease_id === input.disease);

  // 3. If disease not found -> return error
  if (!exactMatch) {
    throw new Error(`Disease '${input.disease}' not found in verified database`);
  }

  // 4. If found -> return linked treatment products
  if (Array.isArray(exactMatch.treatments)) {
    return exactMatch.treatments;
  }

  const productDatabase = products || [];
  const treatmentIds = exactMatch.treatment_ids || [];
  return treatmentIds
    .map(id => productDatabase.find(product => product.product_id === id))
    .filter(Boolean);
}

// Export for node environment testing, attach to window for browser usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { getRecommendation };
} else if (typeof window !== 'undefined') {
  window.getRecommendation = getRecommendation;
}
