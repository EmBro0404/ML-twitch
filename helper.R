library(jsonlite)

make_chunks <- function(messages, chunk_size = 100) {
  
  n <- length(messages)
  
  # Number of complete 100-message chunks
  n_complete <- floor(n / chunk_size)
  
  if (n_complete == 0) {
    return(list())
  }
  
  # Throw away incomplete final chunk
  messages <- messages[
    seq_len(n_complete * chunk_size)
  ]
  
  # Split into groups of 100
  split(
    messages,
    ceiling(seq_along(messages) / chunk_size)
  )
}


prepare_training_data <- function(
    vectors_directory,
    streamers,
    messages_per_chunk = 100
) {
  
  all_chunks <- list()
  all_labels <- c()
  
  # ------------------------------------------
  # Read each streamer
  # ------------------------------------------
  
  for (streamer in streamers) {
    
    file <- file.path(
      vectors_directory,
      paste0(streamer, ".json")
    )
    
    cat("Reading:", file, "\n")
    
    messages <- fromJSON(
      file,
      simplifyVector = FALSE
    )
    
    # Split into groups of 100 messages
    chunks <- make_chunks(
      messages,
      messages_per_chunk
    )
    
    # Streamer label
    # First streamer = 0
    # Second streamer = 1
    # etc.
    label <- match(streamer, streamers) - 1
    
    # Add chunks
    all_chunks <- c(
      all_chunks,
      chunks
    )
    
    # Give every chunk the streamer's label
    all_labels <- c(
      all_labels,
      rep(label, length(chunks))
    )
    
    cat(
      "  Messages:",
      length(messages),
      "\n"
    )
    
    cat(
      "  Complete chunks:",
      length(chunks),
      "\n"
    )
  }
  
  # ------------------------------------------
  # Shuffle chunks
  # ------------------------------------------
  
  set.seed(123)
  
  shuffle <- sample(
    seq_along(all_chunks)
  )
  
  all_chunks <- all_chunks[shuffle]
  all_labels <- all_labels[shuffle]
  
  # ------------------------------------------
  # Turn chunks into 100 model inputs
  # ------------------------------------------
  
  x_train <- lapply(
    seq_len(messages_per_chunk),
    function(message_number) {
      
      lapply(
        all_chunks,
        function(chunk) {
          chunk[[message_number]]
        }
      )
    }
  )
  
  # ------------------------------------------
  # Return
  # ------------------------------------------
  
  list(
    x = x_train,
    y = all_labels
  )
}

