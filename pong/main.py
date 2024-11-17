from pong.models import Session


def main():
    session = Session(
        nb_epochs=1500,
        max_nb_steps=10000, 
        memory_size=10000,
        batch_size=32,
        training=True,
        checkpoint_frequency=100,
    )

    session.run()
    session.close()

if __name__ == "__main__":
    main()

