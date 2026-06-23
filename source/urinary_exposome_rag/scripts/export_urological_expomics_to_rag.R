library(data.table)

source_dir <- "C:/Users/Administrator/Desktop/UrologicalExpomics/data"
output_dir <- "C:/Users/Administrator/Documents/New project 2/urinary_exposome_rag/data/effects"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

as_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

map_disease_group <- function(icd10, disease) {
  icd10 <- as.character(icd10)
  disease <- tolower(as.character(disease))
  ifelse(grepl("^C64|kidney_cancer|renal", icd10) | grepl("kidney cancer|renal", disease), "renal_cancer",
  ifelse(grepl("^C67|bladder", icd10) | grepl("bladder", disease), "bladder_cancer",
  ifelse(grepl("^N17|acute kidney", icd10) | grepl("acute kidney", disease), "acute_kidney_injury",
  ifelse(grepl("^N18|chronic kidney", icd10) | grepl("chronic kidney|ckd", disease), "chronic_kidney_disease",
  ifelse(grepl("^N20|^N21|stone|calculus", icd10) | grepl("stone|calculus", disease), "urolithiasis",
  ifelse(grepl("^N30|^N39|urinary tract infection|uti|cystitis", icd10) | grepl("urinary tract infection|uti|cystitis", disease), "urinary_tract_infection",
  ifelse(grepl("^N40|^R32|^R33|^R35|^R39|luts|nocturia|incontinence|prostatic", icd10) | grepl("luts|nocturia|incontinence|prostatic|benign prostatic", disease), "lower_urinary_tract_symptoms",
         paste0("local_", icd10))))))))
}

map_exposure_domain <- function(category, exposure) {
  text <- tolower(paste(category, exposure))
  ifelse(grepl("air|pm2|pm10|no2|noise|traffic|greenspace|natural environment|road|pollution|pfas|phthalate|phenol|pesticide|metal|cadmium|lead|arsenic|mercury|bisphenol", text), "environmental_pollution",
  ifelse(grepl("sleep|smok|tobacco|alcohol|diet|meat|fish|coffee|tea|water|activity|sedentary|bmi|obesity|supplement|sun|outdoor", text), "lifestyle",
  ifelse(grepl("temperature|climate|heat|humidity|green|built|urban|townsend|deprivation|income|education|employment|housing|household", text), "climate_built_environment",
  ifelse(grepl("diabetes|hypertension|cardiovascular|gout|dyslipidemia|disease|comorbidity", text), "baseline_disease",
         "local_exposomics"))))
}

normalize_result <- function(df, dataset, population, measure) {
  dt <- as.data.table(df)
  if (!("outcome" %in% names(dt))) dt[, outcome := NA_character_]
  if (!("exposure" %in% names(dt))) dt[, exposure := NA_character_]
  if (!("Category" %in% names(dt))) dt[, Category := NA_character_]
  if (!("ICD10" %in% names(dt))) dt[, ICD10 := NA_character_]
  if (!("Disease" %in% names(dt))) dt[, Disease := outcome]
  if (!("exposure0" %in% names(dt))) dt[, exposure0 := exposure]
  if (!("P" %in% names(dt))) dt[, P := NA_real_]
  if (!("FDR" %in% names(dt))) dt[, FDR := NA_real_]
  estimate_col <- if (measure == "OR") "OR" else "HR"
  if (!(estimate_col %in% names(dt))) return(data.table())

  out <- data.table(
    pmid = "",
    pmcid = "",
    title = paste0("Local UrologicalExpomics ", dataset, " result"),
    source_type = "local_urological_expomics",
    source_dataset = dataset,
    population = population,
    outcome = as.character(dt$outcome),
    disease = as.character(dt$Disease),
    exposure = as.character(dt$exposure),
    exposure0 = as.character(dt$exposure0),
    category = as.character(dt$Category),
    measure = measure,
    estimate = as.character(as_num(dt[[estimate_col]])),
    ci_low = as.character(as_num(dt$lower)),
    ci_high = as.character(as_num(dt$upper)),
    p_operator = ifelse(!is.na(as_num(dt$P)), "=", ""),
    p_value = as.character(as_num(dt$P)),
    fdr = as.character(as_num(dt$FDR)),
    icd10 = as.character(dt$ICD10),
    source_url = "",
    doi = ""
  )

  out[, exposure_domains := map_exposure_domain(category, exposure)]
  out[, disease_groups := map_disease_group(icd10, disease)]
  out[, specific_exposure_candidates := exposure]
  out[, china_or_chinese_population_flag := "no"]
  out[, source_location := "local_analysis_table"]
  out[, location_label := paste(dataset, population)]
  out[, snippet := paste0(
    "Dataset: ", dataset,
    "; population: ", population,
    "; exposure: ", exposure,
    "; category: ", category,
    "; outcome: ", outcome,
    "; disease: ", disease,
    "; ", measure, "=", estimate,
    ifelse(!is.na(ci_low) & ci_low != "NA" & !is.na(ci_high) & ci_high != "NA", paste0(" (95% CI ", ci_low, "-", ci_high, ")"), ""),
    ifelse(!is.na(p_value) & p_value != "NA", paste0("; P=", p_value), ""),
    ifelse(!is.na(fdr) & fdr != "NA", paste0("; FDR=", fdr), "")
  )]
  out[, extraction_level := "local_shiny_analysis_result"]
  out[, needs_manual_check := "yes"]
  out[]
}

read_if_exists <- function(path) {
  if (!file.exists(path)) return(NULL)
  readRDS(path)
}

rows <- list()

ukb_files <- list(
  all = "uro.XWAS_resultslistlast.all2.rds",
  male = "uro.XWAS_resultslist.male2.rds",
  female = "uro.XWAS_resultslist.female2.rds"
)
for (population in names(ukb_files)) {
  obj <- read_if_exists(file.path(source_dir, ukb_files[[population]]))
  if (!is.null(obj)) rows[[paste0("UKB_", population)]] <- normalize_result(obj, "UKB exogenous XWAS", population, "HR")
}

nhanes_files <- list(
  all = "uro.XWASnhanes_resultslistlast.all2.rds",
  male = "uro.XWASnhanes_resultslist.male2.rds",
  female = "uro.XWASnhanes_resultslist.female2.rds"
)
for (population in names(nhanes_files)) {
  obj <- read_if_exists(file.path(source_dir, nhanes_files[[population]]))
  if (!is.null(obj)) rows[[paste0("NHANES_", population)]] <- normalize_result(obj, "NHANES endogenous XWAS", population, "OR")
}

weendo <- read_if_exists(file.path(source_dir, "uro.WeEndPd_results2.rds"))
if (!is.null(weendo)) rows[["WeEndPd"]] <- normalize_result(weendo, "WeEndPd preliminary", "all", "HR")

combined <- rbindlist(rows, fill = TRUE)
combined <- combined[!is.na(estimate) & estimate != "NA" & estimate != ""]

csv_path <- file.path(output_dir, "effect_estimates_urological_expomics_local.csv")
jsonl_path <- file.path(output_dir, "effect_estimates_urological_expomics_local.jsonl")
fwrite(combined, csv_path, bom = TRUE)

con <- file(jsonl_path, open = "w", encoding = "UTF-8")
on.exit(close(con), add = TRUE)
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
for (i in seq_len(nrow(combined))) {
  writeLines(jsonlite::toJSON(as.list(combined[i]), auto_unbox = TRUE, na = "null"), con)
}

cat("rows:", nrow(combined), "\n")
cat("csv:", csv_path, "\n")
cat("jsonl:", jsonl_path, "\n")
cat("datasets:\n")
print(combined[, .N, by = .(source_dataset, population)][order(source_dataset, population)])
