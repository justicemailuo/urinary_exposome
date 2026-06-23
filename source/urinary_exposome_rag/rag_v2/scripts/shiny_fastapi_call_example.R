library(httr2)

payload <- list(
  query = "UKB 中 PM2.5 和肾癌的 HR 结果，同时和论文证据分开说",
  top_k = 8,
  use_llm = TRUE,
  filters = list(
    source = "all",
    exposure_domain = "environmental_pollution",
    disease_group = "renal_cancer",
    effects_only = TRUE,
    table_only = FALSE,
    chinese_only = FALSE
  )
)

response <- request("http://127.0.0.1:8890/api/chat") |>
  req_method("POST") |>
  req_body_json(payload, auto_unbox = TRUE) |>
  req_perform()

resp_body_json(response)
