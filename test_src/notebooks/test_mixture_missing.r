library(MixtureMissing)
library(data.table)

test_data = read.table("test_src/test_missing_data.csv",sep=",", header=TRUE)
test_data$X = NULL
head(test_data)

res = MCNM(test_data, G = 2,init_method = "kmedoids", epsilon = 1)
res$clusters

eps = 1e-4
res = NULL
while(is.null(res)){
   res <- tryCatch(
    {
      MCNM(test_data, 
      G = 2, 
      init_method = "kmeans", 
      epsilon = eps, 
      max_iter = 20,progress = FALSE)
    },
    error = function(e) {
      return(NULL)
    }
  )
  eps <- eps * 10
}

results = res$clusters
