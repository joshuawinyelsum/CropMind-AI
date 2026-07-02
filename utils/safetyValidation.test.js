const { validateTreatmentSafety } = require('./safetyValidation');

const validTreatment = {
  product_name: "Amistar Xtra",
  dosage: "30ml per 15L",
  application_steps: ["Step 1", "Step 2"],
  safety_instructions: ["Wear gloves"]
};

const missingDosage = {
  product_name: "Amistar Xtra",
  application_steps: ["Step 1"],
  safety_instructions: ["Wear gloves"]
};

const emptyArray = {
  product_name: "Amistar Xtra",
  dosage: "30ml per 15L",
  application_steps: [], // Empty array
  safety_instructions: ["Wear gloves"]
};

console.log("Running Safety Validation Tests...\n");

// Test 1: Valid Treatment
const res1 = validateTreatmentSafety(validTreatment);
console.assert(typeof res1 === 'object' && res1.product_name === "Amistar Xtra", "Test 1 Failed: Should pass valid treatment unchanged");

// Test 2: Missing Field
const res2 = validateTreatmentSafety(missingDosage);
console.assert(res2 === "Treatment data incomplete. Do not apply.", "Test 2 Failed: Should block missing dosage");

// Test 3: Empty Array
const res3 = validateTreatmentSafety(emptyArray);
console.assert(res3 === "Treatment data incomplete. Do not apply.", "Test 3 Failed: Should block empty application steps");

// Test 4: Null input
const res4 = validateTreatmentSafety(null);
console.assert(res4 === "Treatment data incomplete. Do not apply.", "Test 4 Failed: Should block null input");

console.log("All safety validation tests passed!");
