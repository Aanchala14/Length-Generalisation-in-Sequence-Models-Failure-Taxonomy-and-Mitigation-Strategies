from src.tasks.addition import AdditionTask

task = AdditionTask(sequence_length=2)

sample = task.generate_example()

print(sample)