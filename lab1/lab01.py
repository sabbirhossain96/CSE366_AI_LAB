import random,os

class StudentID:
    
    def __init__(self,path):
        self.path=path
        self.student_id_counter = []
        self.student_id_insert()
      
    # Insert student id function  
    def student_id_insert(self):
        directory= os.path.dirname(self.path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        #Take 120 student in Each Semester with initial id and counter valuw 0.
        if not os.path.exists(self.path):
            with open (self.path,'w') as file:
                for i in range(1,120):
                    student_id = f"2021-1-60-{i:03}"
                    file.write(f"{student_id},0\n")
                for i in range(1,120):
                    student_id = f"2021-2-60-{i:03}"
                    file.write(f"{student_id},0\n")
                for i in range(1,120):
                    student_id = f"2021-3-60-{i:03}"
                    file.write(f"{student_id},0\n")
            print(f"Specific roll numbers generated and saved to {self.path}.")
        
        else:
            print(f"File '{self.path}' already exists. No new rolls generated.")
        
    # Randomly select student id function
    def get_random_student_id(self,students_to_select):
        with open(self.path, 'r') as file:
            self.student_id_counter = [line.strip().split(',') for line in file.readlines()]
            
            available_id = [entry for entry in self.student_id_counter if entry [1] == '0']
            
            if(students_to_select > len(available_id)):
                print("You requested more IDs than are available.")
                return []
            
            selected_id = random.sample(available_id,students_to_select) # Randomly select student.
            print("Before Selected student ID with counter value 0:", [(student_id[0], student_id[1]) for student_id in selected_id]) #Print The selected id with counter value.
            
            for student_id , _ in selected_id:
                for i in range (len(self.student_id_counter)):
                    if self.student_id_counter[i][0] == student_id:
                        counter = int(self.student_id_counter[i][1])+1
                        self.student_id_counter[i][1] = str(counter)
                        
            updated_id = [(student_id[0], student_id[1]) for student_id in selected_id]
            print("After Selected student ID update counter value 1:", updated_id)   #Print The selected id after update counter value.  
                        
            with open(self.path, 'w') as file :
                for student_id , counter in self.student_id_counter:
                    file.write(f"{student_id},{counter}\n")
            return [student_id for student_id, _ in selected_id]
        
    # Reset Function
    def reset_value(self): # Reset counter value to 0 after press 0 which is used for exit  from the code.  
        with open(self.path, 'r') as file :
            self.student_id_counter = [line.strip().split( ',') for line in file.readlines()]
        for i in range (len(self.student_id_counter)):
            self.student_id_counter[i][1] =  '0' 
                 
        with open(self.path, 'w') as file :
            for student_id ,counter in self.student_id_counter:
                file.write(f"{student_id},{counter}\n")
        print(f"All student ID viva counters have been reset to 0 in {self.path}.") 

# Main function     
if __name__ == "__main__":
    
    path="C:\\CSE366-ARTIFICIAL-INTELLIGENCE\\lab1\\student_id.txt"
    run=StudentID(path)
    
    while True:
        number = int(input("How many random student IDs would you like to pick (between 1 and 10)? (Type '0' to quit) "))
        if number == 0:
            run.reset_value()
            print("Exiting the program and resetting student ID counters to 0.")
            break
            
        try:
            if 1<=number<=10:
                selected_id = run.get_random_student_id(number)
            else:
                print("Please Enter Value Between 1 and 10")
        except ValueError:
             print("Invalid input. Please enter a numeric value or '0' to quit.")
