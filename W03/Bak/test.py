# from ast import While
# import json

# json_str = "{'first_name': 'Scott', 'last_name': 'LeFoll', 'food': ['pizza', 'sushi']}"

# my_data = json.loads(json_str)
    
# try:
#     file = open('myfile.json', 'w+') # open or create file for writing and reading
#     try:
#             file.write(json.dumps(my_data))
#             file.close
#     except:
#         print("Error writing json file")
#     finally:
#         file.close()
# except:
#     print("Error opening json file")
    
    # file = open('myfile.json', 'w+') # open or create file for writing and reading
# file.write(json.dumps(my_data))
# file.close

# file = open('myfile.json', 'w+') # open or create file for writing and reading
# data = json.load(file)
# file.close
# print(data)


# Write the Python code for: Put the first 10 prime numbers in an array (list). Display the entire list. Display the first three, last three, and middle three.

prime_limit_int = 0
bad_int = True

print()

try:
    prime_limit_int = int(input("How many primes to find?: "))
except ValueError:
    print("Invalid input. Please enter an integer.")
    print()


# try:

#     it_is = True
# except ValueError:
#     it_is = False

print(it_is)

prime_lst = []

for num in range(2, prime_limit_int + 1):
    for i in range(2, num):
        if (num % i) == 0:
            i = num
    else:
        prime_lst.append(num)
       
print() 
print(prime_lst)
print()
print(prime_lst[:3])
print()
print(prime_lst[-3:])
print()
print(prime_lst[3:6])