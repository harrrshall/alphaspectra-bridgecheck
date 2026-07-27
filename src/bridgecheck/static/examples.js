/*
 * Local, deterministic examples for the BridgeCheck research interface.
 *
 * Names describe spectral geometry or test behavior only. They deliberately
 * carry no disease, health, treatment, water-status, or sensor-equivalence label.
 */

const CABO_MEASURED_CSV = `wavelength_nm,reflectance
400,0.041796360164880753
404,0.041979189962148666
408,0.041905719786882401
412,0.042015288025140762
416,0.042335417121648788
420,0.042702604085206985
424,0.042984191328287125
428,0.043213963508605957
432,0.043195106089115143
436,0.043171938508749008
440,0.043188821524381638
444,0.043383318930864334
448,0.043619900941848755
452,0.043891575187444687
456,0.044052202254533768
460,0.044189956039190292
464,0.044214650988578796
468,0.044205822050571442
472,0.044313900172710419
476,0.044272094964981079
480,0.044333543628454208
484,0.044399634003639221
488,0.044661257416009903
492,0.045041520148515701
496,0.045655939728021622
500,0.046730242669582367
504,0.048460241407155991
508,0.051226526498794556
512,0.055491700768470764
516,0.061797033995389938
520,0.070313751697540283
524,0.080399639904499054
528,0.090312257409095764
532,0.098341882228851318
536,0.10370570421218872
540,0.10700312256813049
544,0.10933524370193481
548,0.11130467057228088
552,0.11235174536705017
556,0.1113259568810463
560,0.10792453587055206
564,0.10251537710428238
568,0.095898926258087158
572,0.088887348771095276
576,0.082490012049674988
580,0.077524401247501373
584,0.073922976851463318
588,0.071361318230628967
592,0.069661401212215424
596,0.06858622282743454
600,0.067550167441368103
604,0.066080592572689056
608,0.063948690891265869
612,0.061471503227949142
616,0.059135641902685165
620,0.057405386120080948
624,0.056426886469125748
628,0.05591738224029541
632,0.055634748190641403
636,0.054979626089334488
640,0.053718786686658859
644,0.052048180252313614
648,0.050520762801170349
652,0.049414191395044327
656,0.048504088073968887
660,0.047595813870429993
664,0.046774379909038544
668,0.046358712017536163
672,0.04646773636341095
676,0.047243118286132812
680,0.048453997820615768
684,0.050110694020986557
688,0.053091064095497131
692,0.060598749667406082
696,0.07745998352766037
700,0.10610321164131165
704,0.14423467218875885
708,0.18721869587898254
712,0.23124504089355469
716,0.27237501740455627
720,0.317097008228302
724,0.35782292485237122
728,0.39427989721298218
732,0.42619505524635315
736,0.45314383506774902
740,0.47467261552810669
744,0.49106210470199585
748,0.50293409824371338
752,0.51124095916748047
756,0.51673334836959839
760,0.52012491226196289
764,0.52212029695510864
768,0.52312886714935303
772,0.52353066205978394
776,0.5235830545425415
780,0.52350771427154541
784,0.52340340614318848
788,0.52323794364929199
792,0.52311277389526367
796,0.52299231290817261
800,0.52287119626998901
804,0.52276968955993652
808,0.52273309230804443
812,0.52273446321487427
816,0.52276086807250977
820,0.52281886339187622
824,0.52283883094787598
828,0.52288824319839478
832,0.52298128604888916
836,0.5230526328086853
840,0.52319270372390747
844,0.52337992191314697
848,0.52350783348083496
852,0.52371758222579956
856,0.52377283573150635
860,0.5239098072052002
864,0.52397310733795166
868,0.52396345138549805
872,0.52394062280654907
876,0.5239107608795166
880,0.5237656831741333
884,0.5236736536026001
888,0.52354663610458374
892,0.52337485551834106
896,0.52316576242446899
900,0.52296704053878784
904,0.52279973030090332
908,0.52263319492340088
912,0.52243989706039429
916,0.52224999666213989
920,0.52195948362350464
924,0.52152091264724731
928,0.52103120088577271
932,0.52038782835006714
936,0.5198369026184082
940,0.51890820264816284
944,0.51808673143386841
948,0.51682144403457642
952,0.51539504528045654
956,0.51386618614196777
960,0.51241302490234375
964,0.51115900278091431
968,0.51031559705734253
972,0.50953638553619385
976,0.50910568237304688
980,0.50846695899963379
984,0.50790274143218994
988,0.50726747512817383
992,0.50665038824081421
996,0.50587964057922363
1000,0.50514459609985352
`;

export const EXAMPLE_DEFINITIONS = Object.freeze([
  Object.freeze({
    id: "measured-cabo",
    label: "Measured CABO",
    shortLabel: "Measured",
    method: "embedded_csv",
    inputOrigin: "measured_training_example",
    provenance: "Anonymized CABO training-group median · not independent validation",
    expectation: "Expected · inside reference-fit range",
  }),
  Object.freeze({
    id: "lower-reference",
    label: "Lower NIR",
    shortLabel: "Bank 1191",
    method: "bank_state",
    stateIndex: 1191,
    inputOrigin: "generated_bank_example_not_measured",
    provenance: "Generated physics-bank reference · not measured",
    expectation: "Expected · exact bank match",
  }),
  Object.freeze({
    id: "median-reference",
    label: "Median NIR",
    shortLabel: "Bank 0083",
    method: "bank_state",
    stateIndex: 83,
    inputOrigin: "generated_bank_example_not_measured",
    provenance: "Generated physics-bank reference · not measured",
    expectation: "Expected · exact bank match",
  }),
  Object.freeze({
    id: "higher-reference",
    label: "Higher NIR",
    shortLabel: "Bank 0470",
    method: "bank_state",
    stateIndex: 470,
    inputOrigin: "generated_bank_example_not_measured",
    provenance: "Generated physics-bank reference · not measured",
    expectation: "Expected · exact bank match",
  }),
  Object.freeze({
    id: "support-warning",
    label: "Support warning",
    shortLabel: "+0.05 test",
    method: "offset_state",
    stateIndex: 470,
    offset: 0.05,
    inputOrigin: "constructed_support_test_not_measured",
    provenance: "Constructed support test · not a biological or sensor example",
    expectation: "Expected · fit warning",
  }),
]);

const EXAMPLES_BY_ID = new Map(EXAMPLE_DEFINITIONS.map((definition) => [definition.id, definition]));

function parseMeasuredExample() {
  const rows = CABO_MEASURED_CSV.trim().split(/\r?\n/).slice(1);
  const wavelengthNm = [];
  const reflectance = [];
  rows.forEach((row) => {
    const [wavelength, value] = row.split(",").map(Number);
    wavelengthNm.push(wavelength);
    reflectance.push(value);
  });
  return { wavelengthNm, reflectance };
}

function assertVerifiedArtifact(artifact) {
  if (!artifact?.bank || !artifact?.shape || !artifact?.contextIndices || !artifact?.wavelengthsNm) {
    throw new Error("a verified BridgeCheck artifact is required for examples");
  }
}

function bankContext(artifact, stateIndex, offset = 0) {
  const [stateCount, bandCount] = artifact.shape;
  if (!Number.isInteger(stateIndex) || stateIndex < 0 || stateIndex >= stateCount) {
    throw new Error(`example state ${stateIndex} is unavailable`);
  }
  const wavelengthNm = [];
  const reflectance = [];
  artifact.contextIndices.forEach((band) => {
    wavelengthNm.push(artifact.wavelengthsNm[band]);
    reflectance.push(artifact.bank[stateIndex * bandCount + band] + offset);
  });
  return { wavelengthNm, reflectance };
}

export function buildExample(artifact, exampleId) {
  assertVerifiedArtifact(artifact);
  const definition = EXAMPLES_BY_ID.get(exampleId);
  if (!definition) {
    throw new Error(`unknown BridgeCheck example: ${exampleId}`);
  }
  let spectrum;
  if (definition.method === "embedded_csv") {
    spectrum = parseMeasuredExample();
  } else if (definition.method === "bank_state") {
    spectrum = bankContext(artifact, definition.stateIndex);
  } else if (definition.method === "offset_state") {
    spectrum = bankContext(artifact, definition.stateIndex, definition.offset);
  } else {
    throw new Error(`unsupported example method: ${definition.method}`);
  }
  return {
    ...definition,
    filename: `example-${definition.id}.csv`,
    wavelengthNm: spectrum.wavelengthNm,
    reflectance: spectrum.reflectance,
  };
}

export function exampleToCsv(example) {
  const lines = ["wavelength_nm,reflectance"];
  example.wavelengthNm.forEach((wavelength, index) => {
    lines.push(`${wavelength},${Number(example.reflectance[index]).toPrecision(17)}`);
  });
  return `${lines.join("\n")}\n`;
}
