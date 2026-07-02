/**
 * Safety Validation Module for Agricultural Treatments
 *
 * Enforces strict presence of critical safety and application fields.
 * No auto-filling or assumptions are made.
 *
 * @param {Object} treatment - The treatment object to validate
 * @returns {Object|string} Returns the unchanged treatment object if valid,
 *                          otherwise returns "Treatment data incomplete. Do not apply."
 */
function validateTreatmentSafety(treatment) {
  if (!treatment || typeof treatment !== 'object') {
    return "Treatment data incomplete. Do not apply.";
  }

  const requiredFields = [
    'dosage',
    'application_steps',
    'safety_instructions'
  ];

  if (!treatment.product_name && !treatment.name) {
    return "Treatment data incomplete. Do not apply.";
  }

  // Strict validation: check if any required field is missing, null, or empty
  for (const field of requiredFields) {
    const value = treatment[field];
    
    if (value === undefined || value === null || value === '') {
      return "Treatment data incomplete. Do not apply.";
    }

    // If it's an array (like application_steps or safety_instructions), ensure it's not empty
    if (Array.isArray(value) && value.length === 0) {
      return "Treatment data incomplete. Do not apply.";
    }
  }

  // If valid: Pass data forward unchanged
  return treatment;
}

// Export for node environment testing, attach to window for browser usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { validateTreatmentSafety };
} else if (typeof window !== 'undefined') {
  window.validateTreatmentSafety = validateTreatmentSafety;
}
