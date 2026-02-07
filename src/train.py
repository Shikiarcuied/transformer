def train_transformer():
    # ======================
    # Hyperparameters
    # ======================
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.0001

    # ======================
    # Model initialization
    # ======================
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    )

    # ======================
    # Loss and optimizer
    # ======================
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # ======================
    # Dummy training data
    # ======================
    src_data = torch.randint(
        1, src_vocab_size, (batch_size, max_seq_length)
    )
    tgt_data = torch.randint(
        1, tgt_vocab_size, (batch_size, max_seq_length)
    )

    # ======================
    # Training loop
    # ======================
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(src_data, tgt_data[:, :-1])

        # Compute loss
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    # ======================
    # Validation
    # ======================
    model.eval()
    val_src = torch.randint(
        1, src_vocab_size, (batch_size, max_seq_length)
    )
    val_tgt = torch.randint(
        1, tgt_vocab_size, (batch_size, max_seq_length)
    )

    with torch.no_grad():
        val_output = model(val_src, val_tgt[:, :-1])
        val_loss = criterion(
            val_output.contiguous().view(-1, tgt_vocab_size),
            val_tgt[:, 1:].contiguous().view(-1),
        )

        print(f"Validation Loss: {val_loss.item()}")


if __name__ == "__main__":
    train_transformer()
