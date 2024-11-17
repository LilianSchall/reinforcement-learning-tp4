from pong.models import Session


def main():
    session = Session(
        nb_epochs=3000,
        max_nb_steps=10000, 
        memory_size=100000,
        batch_size=64,
        training=True,
        checkpoint_frequency=2,
    )

    session.run()
    session.close()

if __name__ == "__main__":
    main()

