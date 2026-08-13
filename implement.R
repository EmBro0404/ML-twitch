library(keras3)
library(reticulate)
library(jsonlite)

source_python("traindata_prepare.py")

tf <- import("tensorflow")

model <- load_model("twitch_streamer_model.keras")

clean_csv_from_twitchchatdownloader.com <- function(csv) {
  df <- read.csv(csv, header=TRUE)
  df <- as.character(df[,4])
  
  temp_file <- tempfile(fileext = ".txt")
  
  writeLines(df, temp_file)
  df <- transform_messages_to_vectors(temp_file,
                                      "lookup.csv",
                                      return_dataset = "true")
  unlink(temp_file)
  
  return(df)
}


predict_input_hundred <- function(hundred_messages) {
  if (length(hundred_messages) < 100) {
    stop("Fewer than 100 messages supplied. The model requires exactly 100 messages.")
  }
  
  if (length(hundred_messages) > 100) {
    message("More than 100 messages supplied. Only the first 100 messages will be used.")
    hundred_messages <- hundred_messages[1:100]
  }

  prediction_input <- lapply(hundred_messages, function(x) {
    tf$ragged$constant(
      list(as.list(as.integer(x))),
      dtype = tf$int32
    )
  })


prediction <- predict(
  model,
  prediction_input
)

print('
  cinna 0,
  dantes 1,
  extraemily 2,
  ishowspeed 3,
  jasontheween 4,
  ludwig 5,
  marlon 6,
  moistcr1tikal 7,
  xqc 8
      ')

print(prediction)

}

predict_twitch <- function(predict_file = "predict.csv") {
predict_input_hundred(clean_csv_from_twitchchatdownloader.com(predict_file))
}


