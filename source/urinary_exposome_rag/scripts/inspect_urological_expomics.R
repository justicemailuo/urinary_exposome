cat("R version:", R.version.string, "\n")

packages <- c("shiny", "jsonlite", "httr2", "arrow", "data.table", "dplyr", "DT")
for (pkg in packages) {
  cat(pkg, ":", requireNamespace(pkg, quietly = TRUE), "\n")
}

base_dir <- "C:/Users/Administrator/Desktop/UrologicalExpomics/data"
files <- list.files(base_dir, full.names = TRUE)
cat("files:", length(files), "\n")

print_head <- function(x, n = 2) {
  print(utils::head(x, n))
}

for (path in files) {
  cat("\n---", basename(path), file.info(path)$size, "\n")
  if (grepl("\\.csv$", path, ignore.case = TRUE)) {
    x <- try(read.csv(path, nrows = 3, check.names = FALSE), silent = TRUE)
    if (!inherits(x, "try-error")) {
      print(names(x))
      print_head(x)
    } else {
      cat("csv read error\n")
    }
  } else if (grepl("\\.rds$", path, ignore.case = TRUE)) {
    x <- try(readRDS(path), silent = TRUE)
    if (!inherits(x, "try-error")) {
      cat("class:", paste(class(x), collapse = ","), "\n")
      if (is.data.frame(x)) {
        cat("dim:", paste(dim(x), collapse = "x"), "\n")
        print(names(x)[seq_len(min(30, ncol(x)))])
        print_head(x)
      } else if (is.list(x)) {
        cat("list length:", length(x), "\n")
        print(names(x)[seq_len(min(30, length(x)))])
        if (length(x) > 0 && is.data.frame(x[[1]])) {
          cat("first element class/dim:", paste(class(x[[1]]), collapse = ","), paste(dim(x[[1]]), collapse = "x"), "\n")
          print(names(x[[1]])[seq_len(min(30, ncol(x[[1]])))])
          print_head(x[[1]])
        }
      } else {
        str(x, max.level = 1)
      }
    } else {
      cat("rds read error\n")
    }
  } else if (grepl("\\.parquet$", path, ignore.case = TRUE) && requireNamespace("arrow", quietly = TRUE)) {
    dataset <- try(arrow::open_dataset(path), silent = TRUE)
    if (!inherits(dataset, "try-error")) {
      print(dataset$schema)
      y <- try(as.data.frame(utils::head(dataset, 3)), silent = TRUE)
      if (!inherits(y, "try-error")) {
        print(names(y)[seq_len(min(30, ncol(y)))])
        print(y[, seq_len(min(8, ncol(y))), drop = FALSE])
      }
    } else {
      cat("parquet read error\n")
    }
  }
}
