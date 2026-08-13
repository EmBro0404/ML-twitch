library(keras3)
library(reticulate)
library(jsonlite)

source("helper.R")

"
install_keras(
  backend = 'tensorflow',
  gpu = TRUE
)
"


# ============================================================
# Prepare data to session (~6 GB)
# ============================================================

streamers <- c(
  "0",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8"
)


# save files to to R session
training_data <- prepare_training_data(paste0(getwd(), "/messages"),
                      streamers,
                      messages_per_chunk = 100
                      )
for (i in seq_along(training_data$x)) {
  names(training_data$x[[i]]) <- NULL
  
  for (j in seq_along(training_data$x[[i]])) {
    names(training_data$x[[i]][[j]]) <- NULL
  }
}

tf <- import("tensorflow")

# Convert to tensorflow object
for (i in seq_along(training_data$x)) {
  training_data$x[[i]] <- tf$ragged$constant(
    training_data$x[[i]],
    dtype = tf$int32
  )
}


# ============================================================
# SETTINGS
# ============================================================

vocab_size <- 203535 + 1 # (+1 for the 0 for unidentified words)
embedding_dim <- 64
messages_per_chunk <- 100 # for the training data, the input in implementation can be any size
num_streamers <- 9
validation_split = 0.2


# ============================================================
# SHARED TRAINABLE EMBEDDING
# ============================================================
#
# This is ONE matrix:
#
# vocab_size × embedding_dim
#
# Every one of the 100 messages uses this SAME matrix.
#
# It is trainable and will be updated by backpropagation.

embedding <- layer_embedding(
  input_dim = vocab_size,
  output_dim = embedding_dim,
  name = "word_embedding"
)


# ============================================================
# 100 INPUTS
# ============================================================
#
# Each input is ONE message.
#
# One word is one integer.
#
# Each integer is then turned into a 64 dimensional vector.
#
# The message can contain any number of tokens (any length of words)
#
# Example:
#
# message 1: [17, 42]
# message 2: [91, 12, 55]
# message 3: [31]
#
# No padding.
# No maximum message length.
#
# ragged = TRUE tells Keras that the token sequence length
# can vary between examples.

inputs <- lapply(seq_len(messages_per_chunk), function(i) {
  keras_input(
    shape = c(NA),
    dtype = "int32",
    ragged = TRUE,
    name = paste0("message_", i)
  )
})


# ============================================================
# PROCESS EACH MESSAGE
# ============================================================
#
# Every message uses the SAME embedding layer.
#
# Each message:
#
# token IDs
# ↓
# embedding lookup
# ↓
# N × 64
# ↓
# average over N
# ↓
# 64
#
# The number of tokens N can be different for every message.

message_vectors <- lapply(inputs, function(x) {
  x |>
    embedding() |>
    layer_global_average_pooling_1d()
})


# ============================================================
# COMBINE THE 100 MESSAGE VECTORS
# ============================================================
#
# Each message is now:
#
# 64
#
# Therefore:
#
# 100 × 64
#
# We concatenate them into:
#
# 6400
#
# The order is preserved:
#
# message 1 → first 64 numbers
# message 2 → next 64 numbers
# ...
# message 100 → final 64 numbers

x <- layer_concatenate(
  message_vectors,
  axis = -1
)


# ============================================================
# NORMAL ANN
# ============================================================

outputs <- x |>
  layer_dense(
    units = 256,
    activation = "relu",
    name = "dense_256"
  ) |>
  layer_dense(
    units = 128,
    activation = "relu",
    name = "dense_128"
  ) |>
  layer_dense(
    units = num_streamers,
    activation = "softmax",
    name = "classifier"
  )



# ============================================================
# CREATE MODEL
# ============================================================

model <- keras_model(
  inputs = inputs,
  outputs = outputs
)


# ============================================================
# COMPILE
# ============================================================

model |> compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)


# ============================================================
# INSPECT
# ============================================================

model |> summary()


history <- model |> fit(
  x = training_data$x,
  y = training_data$y,
  epochs = 10,
  batch_size = 32,
  #batch_size = 64,
  #batch_size = 128,
  validation_split = 0.2,
  verbose = 2
)

#get_weights(model)

save_model(model, "twitch_streamer_model.keras")





