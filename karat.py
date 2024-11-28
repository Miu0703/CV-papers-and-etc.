'''
Longest Common Continuous Subarray
输入：


[
  ["3234.html", "xys.html", "7hsaa.html"], // user1
  ["3234.html", "sdhsfjdsh.html", "xys.html", "7hsaa.html"] // user2
]
输出两个user的最长连续且相同的访问记录。
'''
def longestCommonSubarray(user1:list,user2:list)->list:
    n,m=len(user1),len(user2)

    dp=[[0]*(m+1) for _ in range(n+1)]
    max_len=0
    end_pos=0

    for i in range(1,n+1):
        for j in range(1,m+1):
            if user1[i-1]==user2[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
                if dp[i][j]>max_len:
                    max_len=dp[i][j]
                    end_pos=i

    return user1[end_pos-max_len:end_pos]    
'''
A count-paired domain is a domain that has one of the two formats "rep d1.d2.d3" or 
"rep d1.d2" where rep is the number of visits to the domain and d1.d2.d3 is the domain itself.

For example, "9001 discuss.leetcode.com" is a count-paired domain that indicates that 
discuss.leetcode.com was visited 9001 times.
Given an array of count-paired domains cpdomains, return an array of the count-paired domains
 of each subdomain in the input. You may return the answer in any order.'''    
  
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        ans=collections.Counter()
        for domain in cpdomains:
            count,domain=domain.split()
            count=int(count)
            subdomain=domain.split('.')
            for i in range(len(subdomain)):
                ans['.'.join(subdomain[i:])]+=count

        return ["{} {}".format(count,it) for it,count in ans.items()]  
'''Two Sum'''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap:
                return [i, hashmap[complement]]
            hashmap[nums[i]] = i
        # Return an empty list if no solution is found
        return []                 

'''
// The people who buy ads on our network don't have enough data about how ads are working for
//their business. They've asked us to find out which ads produce the most purchases on their website.

// Our client provided us with a list of user IDs of customers who bought something on a landing page
//after clicking one of their ads:

// # Each user completed 1 purchase.
// completed_purchase_user_ids = [
//   "3123122444","234111110", "8321125440", "99911063"]

// And our ops team provided us with some raw log data from our ad server showing every time a
//user clicked on one of our ads:
// ad_clicks = [
//  #"IP_Address,Time,Ad_Text",
//  "122.121.0.1,2016-11-03 11:41:19,Buy wool coats for your pets",
//  "96.3.199.11,2016-10-15 20:18:31,2017 Pet Mittens",
//  "122.121.0.250,2016-11-01 06:13:13,The Best Hollywood Coats",
//  "82.1.106.8,2016-11-12 23:05:14,Buy wool coats for your pets",
//  "92.130.6.144,2017-01-01 03:18:55,Buy wool coats for your pets",
//  "92.130.6.145,2017-01-01 03:18:55,2017 Pet Mittens",
//]
       
//The client also sent over the IP addresses of all their users.
       
//all_user_ips = [
//  #"User_ID,IP_Address",
//   "2339985511,122.121.0.155",
//  "234111110,122.121.0.1",
//  "3123122444,92.130.6.145",
//  "39471289472,2001:0db8:ac10:fe01:0000:0000:0000:0000",
//  "8321125440,82.1.106.8",
//  "99911063,92.130.6.144"
//]
       
// Write a function to parse this data, determine how many times each ad was clicked,
//then return the ad text, that ad's number of clicks, and how many of those ad clicks
//were from users who made a purchase.


// Expected output:
// Bought Clicked Ad Text
// 1 of 2  2017 Pet Mittens
// 0 of 1  The Best Hollywood Coats
// 3 of 3  Buy wool coats for your pets        
'''
from collections import defaultdict

def analyze_ad_clicks(completed_purchase_user_ids, ad_clicks, all_user_ips):
    ip_to_user={}
    for record in all_user_ips:
        user_id,ip_address=record.split(",")
        ip_to_user[ip_address]=user_id


    ad_clicks_count=defaultdict(int)
    ad_purchased_count=defaultdict(int)

    for record in ad_clicks:
        user_id,_,ad_text=record.split(",")
        user_id=ip_to_user.get(ip_address)

        ad_clicks_count[ad_text]+=1

        if user_id in completed_purchase_user_ids:
            ad_purchased_count[ad_text]+=1

    result=[]
    for ad_text in ad_clicks_count:
        total_clicks=ad_clicks_count[ad_text]
        total_purchses=ad_purchased_count[ad_text]
        result.append(f"{total_purchses} of {total_clicks} {ad_text}")

    return result  
'''
Design a hit counter which counts the number of hits received in the past 5 minutes (i.e., the past 300 seconds).

Your system should accept a timestamp parameter (in seconds granularity), and you may assume that calls are being made to the system in chronological order (i.e., timestamp is monotonically increasing). Several hits may arrive roughly at the same time.

Implement the HitCounter class:

HitCounter() Initializes the object of the hit counter system.
void hit(int timestamp) Records a hit that happened at timestamp (in seconds). Several hits may happen at the same timestamp.
int getHits(int timestamp) Returns the number of hits in the past 5 minutes from timestamp (i.e., the past 300 seconds).'''
from collections import deque

class HitCounter:
    """
    HitCounter maintains a queue of timestamps. 
    The `hit()` method appends a new timestamp to the queue.
    The `getHits()` method removes timestamps that are outside the 300-second window 
    and returns the number of remaining timestamps in the queue.
    """
    def __init__(self):
        # Initialize an empty deque to store the timestamps of hits
        self.queue = deque()

    def hit(self, timestamp: int) -> None:
        # Append the given timestamp to the queue
        self.queue.append(timestamp)

    def getHits(self, timestamp: int) -> int:
        # Remove timestamps that are not within the last 300 seconds
        while self.queue and timestamp - self.queue[0] >= 300:
            self.queue.popleft()
        # The remaining timestamps in the queue represent valid hits
        return len(self.queue)


'''
You are a developer for a university. Your current project is to develop a system 
for students to find courses they share with friends. The university has a system for 
querying courses students are enrolled in, returned as a list of (ID, course) pairs.
Write a function that takes in a list of (student ID number, course name) pairs and 
returns, for every pair of students, a list of all courses they share.

'''
from collections import defaultdict
from itertools import combinations

def find_pairs(student_course_pairs):
    student_to_courses=defaultdict(set)
    for student,course in student_course_pairs:
        student_to_courses[student].add(course)
    result={}
    student=list(student_to_courses.keys())
    for student1,student2 in combinations(student,2):
        shared_courses=student_to_courses[student1]&student_to_courses[student2]
        result[(int(student1),int(student2))]=list(shared_courses)

    return result    

'''
Students may decide to take different "tracks" or sequences of courses in the 
Computer Science curriculum. There may be more than one track that includes the same course,
but each student follows a single linear track from a "root" node to a "leaf" node. 
In the graph below, their path always moves left to right.

Write a function that takes a list of (source, destination) pairs, and returns the name
 of all of the courses that the students could be taking when they are halfway through 
 their track of courses.

'''
from collections import defaultdict

def find_mid_courses_dfs(all_courses):
    graph=defaultdict(list)
    in_degree=defaultdict(int)
    for src,dest in all_courses:
        graph[src].append(dest)
        in_degree[dest]+=1
        if src not in in_degree:
            in_degree[src]=0

    roots=[node for node in in_degree if in_degree[node]==0]
    mid_courses=set()

    def dfs(node,path):
        path.append(node)

        if not graph[node]:
            mid=len(path)//2
            mid_courses.add(path[mid])
        else:
            for neighbor in graph[node]:
                dfs(neighbor,path)   
        path.pop() 



    for root in roots:
        dfs(root,[])

    return list(mid_courses)                
##############
from collections import defaultdict,deque
def find_mid_courses_kahn(all_courses):
    graph=defaultdict(list)
    in_degree=defaultdict(int)

    for src, dest in all_courses:
        graph[src].append(dest)
        in_degree[dest]+=1

        if src not in in_degree:
            in_degree[src]=0

    roots=[node for node in in_degree if in_degree[node]==0]

    mid_courses=set()
    for root in roots:
        queue=deque([(root,[])])

        while queue:
            node,path=queue.popleft()
            path.append(node)
            if not graph[node]:
                mid=len(path)//2
                mid_courses.add(path[mid])
            else:
                for neighbor in graph[node]:
                    queue.append((neighbor,path[:]))
    return list(mid_courses)
'''
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you 
must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.'''
from collections import defaultdict, deque

class Solution:
    def canFinish(self, numCourses, prerequisites):
        # 使用 defaultdict 替代嵌套列表
        adj = defaultdict(list)
        indegree = [0] * numCourses

        # 构建邻接表和入度数组
        for prerequisite in prerequisites:
            adj[prerequisite[1]].append(prerequisite[0])
            indegree[prerequisite[0]] += 1

        # 初始化队列，添加所有入度为 0 的节点
        queue = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)

        nodesVisited = 0

        # 拓扑排序结果列表
        #order = []

        # 拓扑排序
        while queue:
            node = queue.popleft()
            nodesVisited += 1
            #order.append(node)  # 将节点加入结果列表

            for neighbor in adj[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        # 如果访问的节点数等于课程总数，则可以完成所有课程
        return nodesVisited == numCourses
        # 如果访问的节点数等于课程总数，则返回拓扑排序结果
        #if len(order) == numCourses:
        #    return order
        #else:
        #    # 否则存在环，返回空数组
        #    return []

'''
Solution 1: Word Wrap with Hyphens
Wrap the words in the list to ensure that the total 
length of each line does not exceed max_length. Words are joined with hyphens.
'''
def wrap_words(word_list, max_length):
    result = []
    current_line = []
    current_length = 0

    for word in word_list:
        if current_length + len(word) + len(current_line) <= max_length:
            current_line.append(word)
            current_length += len(word)
        else:
            result.append("-".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        result.append("-".join(current_line))
    return result


'''
Solution 2: Justify Wrapped Lines
This builds on the previous solution 
but adjusts each line to be fully justified by adding spaces evenly between the words.
'''
def justify_wrap(word_list, max_length):
    def justify_line(line):
        if len(line)==1:
            return line[0]+"-"*(max_length-len(line[0]))
        total_spaces=max_length-sum(len(word) for word in line)
        space_between=total_spaces//(len(line)-1)
        extra_space=total_spaces%(len(line)-1)
        justified=""
        for i,word in enumerate(line[:-1]):
            justified+=word+"-"*(space_between+ (1 if i<extra_space else 0))
        justified += line[-1]    
        return justified    

    result = []
    current_line = []
    current_length = 0

    for word in word_list:
        if current_length + len(word) + len(current_line) <= max_length:
            current_line.append(word)
            current_length += len(word)
        else:
            result.append(justify_line(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        result.append(justify_line(current_line))
    return result



'''
Solution 3: Minimize Balance Score
This solution minimizes the balance score by testing different ways of wrapping words and choosing the arrangement with the lowest score.
'''
from itertools import combinations

def balanced_wrap_lines(text, max_length):
    words = text.split()
    
    def calculate_score(lines):
        max_len = max(len(line) for line in lines)
        return sum((max_len - len(line))**2 for line in lines)
    
    def generate_wraps():
        all_wraps = []  # 用于存储所有可能的换行方案
        for indices in combinations(range(1, len(words)), len(words) - 1):
            lines = []
            prev = 0
            valid = True  # 标记当前方案是否有效
            for idx in indices:
                line = "-".join(words[prev:idx])
                if len(line) > max_length:  # 如果当前行超过 max_length，标记为无效
                    valid = False
                    break
                lines.append(line)
                prev = idx
            if not valid:
                continue  # 跳过当前方案
            # 处理最后一行
            line = "-".join(words[prev:])
            if len(line) <= max_length:
                lines.append(line)
                all_wraps.append(lines)  # 保存当前方案
        return all_wraps


    best_score = float("inf")
    best_wrap = None

    # 将所有换行方案生成为列表
    all_wraps = generate_wraps()

    for wrap in all_wraps:
        score = calculate_score(wrap)
        if score < best_score:
            best_score = score
            best_wrap = wrap

    return best_wrap
'''Find Words That Can Be Formed by Characters
You are given an array of strings words and a string chars.
A string is good if it can be formed by characters from 
chars (each character can only be used once).
Return the sum of lengths of all good strings in words.'''
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
      
        freq={}
        for i in chars:
            if i in freq:
                freq[i]+=1
            else:
                freq[i]=1
        count=0

        for word in words:
            temp=freq.copy()
            flag=True
            for letter in word:
                if letter in temp:
                    if temp[letter]==0:
                        flag=False
                        break
                    temp[letter]-=1
                else:
                    flag=False
                    break

            if flag:
                count+=len(word)

        return count         
'''
Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using 
the letters from magazine and false otherwise.
Each letter in magazine can only be used once in ransomNote.''' 
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        if len(ransomNote)>len(magazine):
            return False
        ransomNote=sorted(ransomNote,reverse=True)
        magazine=sorted(magazine,reverse=True)

        while ransomNote and magazine:
            if ransomNote[-1]==magazine[-1]:
                ransomNote.pop()
                magazine.pop()

            elif magazine[-1]<ransomNote[-1]:
                magazine.pop()
            else:
                return False

        return not ransomNote    
'''
Given two integer arrays nums1 and nums2, return the maximum length of a subarray 
that appears in both arrays.'''   
class Solution(object):
    def findLength(self, A, B):
        ans = 0
        Bstarts = collections.defaultdict(list)
        for j, y in enumerate(B):
            Bstarts[y].append(j)
        for i, x in enumerate(A):
            for j in Bstarts[x]:
                k = 0
                while i + k < len(A) and j + k < len(B) and A[i + k] == B[j + k]:
                    k += 1
                ans = max(ans, k)
        return ans
######################
class Solution(object):
    def findLength(self, A, B):
        # Initialize a DP table with dimensions (len(A)+1) x (len(B)+1)
        # Each cell memo[i][j] represents the longest common prefix of A[i:] and B[j:]
        memo = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
        
        # Iterate over A and B in reverse order
        for i in range(len(A) - 1, -1, -1):
            for j in range(len(B) - 1, -1, -1):
                # If characters match, extend the length of the common prefix
                if A[i] == B[j]:
                    memo[i][j] = memo[i + 1][j + 1] + 1
        
        # Find and return the maximum value in the DP table
        return max(max(row) for row in memo)


'''An n x n matrix is valid if every row and every column 
contains all the integers from 1 to n (inclusive).
Given an n x n integer matrix matrix, return true if the 
matrix is valid. Otherwise, return false.'''
class Solution:
    def checkValid(self, matrix: list[list[int]]) -> bool:
        set_=set(range(1,len(matrix)+1))
        return all(set_==set(x) for x in matrix+list(zip(*matrix)))

'''Given an m x n grid of characters board and a string word,
return true if word exists in the grid.
The word can be constructed from letters of sequentially 
adjacent cells, where adjacent cells are horizontally or 
vertically neighboring. The same letter cell may not be used
 more than once'''
class Solution:
    def exist(self, board, word):
        def backtrack(i, j, k):
            if k == len(word):
                return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
                return False
            
            temp = board[i][j]
            board[i][j] = ''
            
            if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
                return True
            
            board[i][j] = temp
            return False
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if backtrack(i, j, 0):
                    return True
        return False                  
'''
Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells
 are horizontally or vertically neighboring. The same letter cell may not be used more than once 
 in a word.'''
class Solution:
    def findWords(self, board, words):
        # 构建字典树 (Trie)
        def build_trie(words):
            trie = {}
            for word in words:
                node = trie
                for char in word:
                    if char not in node:
                        node[char] = {}
                    node = node[char]
                node["#"] = True  # "#" 表示单词结束
            return trie

        # 回溯搜索单词
        def backtrack(i, j, node, path):
            char = board[i][j]
            if char not in node:  # 不在字典树中，剪枝
                return

            node = node[char]
            path += char
            if "#" in node:  # 找到单词
                result.add(path)

            # 标记当前位置为已访问
            temp = board[i][j]
            board[i][j] = ""

            # 搜索四个方向
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < len(board) and 0 <= nj < len(board[0]) and board[ni][nj]:
                    backtrack(ni, nj, node, path)

            # 恢复当前位置
            board[i][j] = temp

        # 构建字典树
        trie = build_trie(words)

        result = set()
        for i in range(len(board)):
            for j in range(len(board[0])):
                backtrack(i, j, trie, "")

        return list(result)
'''
there is an image filled with 0s and 1s. There is at most one rectangle in this image 
filled with 0s, find the rectangle. Output could be the coordinates of top-left and 
bottom-right elements of the rectangle, or top-left element, width and height.

'''
def find_rectangle(image):
    rows=len(image)
    cols=len(image[0])

    top_left=None
    bottom_right=None

    for i in range(rows):
        for j in range(cols):
            if image[i][j]==0:
                top_left=(i,j)
                break
        if top_left:
            break
    if not top_left:
        return None

    bottom_x,bottom_y=top_left
    for x in range(top_left[0],rows):
        if image[x][top_left[1]]==1:
            break
    bottom_x=x-1
        

    for y in range(top_left[1],cols):
        if image[top_left[0]][y]==1:
            break
    bottom_y=y-1
    bottom_right=(bottom_x,bottom_y)
    return{"top_left":top_left, "bottom_right": bottom_right}             

'''
for the same image, it is filled with 0s and 1s. It may have multiple rectangles filled with 0s. The rectangles are separated by 1s. Find all the rectangles.

'''
def find_all_rectangles(image):
    rows=len(image)
    cols=len(image[0])
    visited=[[False]*cols for _ in range(rows)]
    rectangles=[]

    for i in range(rows):
        for j in range(cols):
            if image[i][j]==0 and not visited[i][j]:
                top_left=(i,j)
                bottom_x,bottom_y=i,j

                while bottom_y+1<cols and image[i][bottom_y+1]==0:
                    bottom_y+=1
                while bottom_x+1<rows and all(image[bottom_x+1][k]==0 for k in range(j,bottom_y+1)) :
                    bottom_x+=1

                for x in range(i,bottom_x+1):
                    for y in range(j,bottom_y+1):
                        visited[x][y]=True

                rectangles.append({"top_left":top_left, "bottom_right":(bottom_x,bottom_y)})
    return rectangles        

'''
for rect in result:
    print(f"Top-left: {rect['top_left']}, Bottom-right: {rect['bottom_right']}")
'''
#number of islands
class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        if not grid:
            return 0


        def dfs(grid,r,c):
            if(
                r<0
                or r>=len(grid)
                or c<0
                or c>=len(grid[0])
                or grid[r][c]!="1"
            ):
                return
            grid[r][c]="0"

            dfs(grid,r+1,c)
            dfs(grid,r-1,c)
            dfs(grid,r,c+1)
            dfs(grid,r,c-1)  

        num_islands=0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=="1":
                    dfs(grid,i,j)
                    num_islands+=1

        return num_islands                  
#给输入为string，例如"2+3-999"，之包含+-操作，返回计算结果
def calculate(expression: str) -> int:
    sign=1
    total=0
    num=0

    for char in expression:
        if char.isdigit():
            num=num*10+int(char)
        elif char=='+':
            total+=num*sign
            num=0
            sign=1

        elif char=='-':
            total+=num*sign
            num=0
            sign=-1
    total+=num*sign
    return total        
              

#加上parenthesis， 例如"2+((8+2)+(3-999))"，返回计算结果
def calculate_with_pare(expression: str) -> int:
    sign=1
    total=0
    num=0
    stack=[]

    for char in expression:
        if char.isdigit():
            num=num*10+int(char)
        elif char=='+':
            total+=num*sign
            num=0
            sign=1

        elif char=='-':
            total+=num*sign
            num=0
            sign=-1
        elif char=='(':
            stack.append(total)
            stack.append(sign)
            total=0
            sign=1
        elif char==')':
            total+=sign*num 
            total*=stack.pop()
            total+=stack.pop()
            num=0    

    total+=num*sign
    return total

'''
给一个N*N的矩阵，判定是否是有效的矩阵。
有效矩阵的定义是每一行或者每一列的数字都必须正好是1到N的数。
输出一个bool。'''
def is_valid_matrix(matrix):
    n=len(matrix)
    valid_set=set(range(1,n+1))

    for row in matrix:
        if set(row)!=valid_set:
            return False
        
    for col in range(n):
        col_set=set(matrix[r][col] for r in range(n))
        if col_set!=valid_set:
            return False
        
    return True    

'''
"""
A nonogram is a logic puzzle, similar to a crossword, in which the player is given a blank grid and has to color it according to some instructions. Specifically, each cell can be either black or white, which we will represent as 0 for black and 1 for white.

+------------+
| 1  1  1  1 |
| 0  1  1  1 |
| 0  1  0  0 |
| 1  1  0  1 |
| 0  0  1  1 |
+------------+

For each row and column, the instructions give the lengths of contiguous runs of black (0) cells. For example, the instructions for one row of [ 2, 1 ] indicate that there must be a run of two black cells, followed later by another run of one black cell, and the rest of the row filled with white cells.

These are valid solutions: [ 1, 0, 0, 1, 0 ] and [ 0, 0, 1, 1, 0 ] and also [ 0, 0, 1, 0, 1 ]
This is not valid: [ 1, 0, 1, 0, 0 ] since the runs are not in the correct order.
This is not valid: [ 1, 0, 0, 0, 1 ] since the two runs of 0s are not separated by 1s.

Your job is to write a function to validate a possible solution against a set of instructions. Given a 2D matrix representing a player's solution; and instructions for each row along with additional instructions for each column; return True or False according to whether both sets of instructions match.

'''
def validateNonogram(matrix, rows, columns):
    def validate_line(line,rule):
        lengths=[]
        count=0
        for cell in line:
            if cell==0:
                count+=1
            elif count>0:
                lengths.append(count)
                count=0
        if count>0:
            lengths.append(count)
        return lengths==rule                
    for row, rule in zip(matrix,rows):
        if not validate_line(row,rule):
            return False
    for col_idx,rule in enumerate(columns):
        column=[matrix[row_idx][col_idx] for row_idx in range(len(matrix))]
        if not validate_line(column,rule):
            return False
    return True               

'''
输入是int[][] input, input[0]是input[1] 的parent，
第一问是只有0个parents和只有1个parent的节点
'''
def find_nodes_with_parents(input):
    in_degree={}
    for parent,child in input:
        if parent not in in_degree:
            in_degree[parent]=0
        if child not in in_degree:
            in_degree[child]=0
        
        in_degree[child]+=1

    zero_parent=[]
    one_parent=[]

    for node,count in in_degree.items():
        if count==0:
            zero_parent.append(node)
        if count==1:
            one_parent.append(node)    
    return zero_parent,one_parent


'''
我们需要解决的问题是：在一组有向图的边（edges）中，
判断两个节点 x 和 y 是否有共同的祖先。
输入
edges：一个二维数组，表示有向图的边，每条边是 [parent, child] 的形式。
x 和 y：两个节点，需要判断它们是否有共同的祖先。
'''
def has_common_ancestor(edges, x, y):
    if not edges:
        return False

    direct_parent={}
    for parent,child in edges:
        if child not in direct_parent:
            direct_parent[child]=set()
        direct_parent[child].add(parent)    
    
    def find_all_parents(node):
        ancestors=set()
        stack=[node]
        while stack:
            curr =stack.pop()
            if curr in direct_parent:
                for parent in direct_parent[curr]:
                    if parent not in ancestors:
                        ancestors.add(parent)
                        stack.append(parent)

        return ancestors

    parent_of_x=find_all_parents(x)
    parent_of_y=find_all_parents(y)
    

    return not parent_of_x.isdisjoint(parent_of_y)   
'''
Given a binary tree, find the lowest common ancestor (LCA)
 of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common 
ancestor is defined between two nodes p and q as the lowest node in T
 that has both p and q as descendants (where we allow a node to be a descendant 
 of itself).”'''  
# 定义 TreeNode 类
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def build_tree(values):
    if not values or values[0] is None:
        return None
    root=TreeNode(values[0])
    queue=[root]
    i=1
    while queue and i<len(values):
        current=queue.pop(0)#pop第一个元素
        if i < len(values) and values[i] is not None:
            current.left=TreeNode(values[i])
            queue.append(current.left)

        i+=1
        if i<len(values) and values[i] is not None:
            current.right=TreeNode(values[i])
            queue.append(current.right)
        i+=1
    return root


class Solution:

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:

        # Stack for tree traversal
        stack = [root]

        # Dictionary for parent pointers
        parent = {root: None}

        # Iterate until we find both the nodes p and q
        while p not in parent or q not in parent:

            node = stack.pop()

            # While traversing the tree, keep saving the parent pointers.
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        # Ancestors set() for node p.
        ancestors = set()

        # Process all ancestors for node p using parent pointers.
        while p:
            ancestors.add(p)
            p = parent[p]

        # The first ancestor of q which appears in
        # p's ancestor set() is their lowest common ancestor.
        while q not in ancestors:
            q = parent[q]
        return q

'''
我们需要找到一个节点的最远祖先（Earliest Ancestor）。具体要求如下：

输入为 parentChildPairs，表示一个有向图的边，
其中 [parent, child] 表示 parent 是 child 的直接父节点。
给定节点 x，要求找到距离它最远的祖先节点。
最远祖先是指，从 x 开始往上追溯，无法再进一步追溯的祖先。
如果某节点没有父节点，则它本身是最远祖先。'''
from collections import defaultdict,deque
def earliest_ancestor(parent_child_pairs,x):
    direct_parents=defaultdict(set)
    for parent, child in parent_child_pairs:
        direct_parents[child].add(parent)

    curr_layer=deque([x])
    visited=set()
    prev_layer=None

    while curr_layer:
        prev_layer=list(curr_layer)
        for _ in range(len(curr_layer)):
            curr=curr_layer.popleft()
            parents=direct_parents.get(curr)
            if not parents:
                continue
            for parent in parents:
                if parent not in visited:
                    curr_layer.append(parent)
                    visited.add(parent)
    if not prev_layer:
        return x
    return prev_layer[0]                      

'''
我们有一组记录，表示人进入或离开大楼的行为。每条记录包括名字和动作（"enter" 或 "exit"）。需要找到以下两类人：

进出记录不符的人：

未带工牌进入大楼（即出现多次连续 "enter" 或最后没有 "exit"）。
未带工牌离开大楼（即出现多次连续 "exit" 或没有 "enter" 就 "exit"）。
输入和输出：

输入：一个二维数组 badge_records，其中每条记录是 [名字, 动作]。
输出：两个列表：
第一个列表是所有 未带工牌进入大楼的人。
第二个列表是所有 未带工牌离开大楼的人。'''
def invalid_badge_records(records):
    if not records:
        return [],[]

    invalid_enter=set()
    invalid_exit=set()

    state={}

    for name,action in records:
        if name not in state:
            state[name]=0

        if action=="enter":
            if state[name]==0:
                state[name]=1
            else:
                invalid_enter.add(name)
        if action=="exit":
            if state[name]==1:
                state[name]=0
            else:
                invalid_exit.add(name)
    for name,s in state.items():
        if s==1:
            invalid_enter.add(name)

    return list(invalid_enter),list(invalid_exit)                                 

'''一小时内access多次

给 list of [name, time], time is string format: '1300' //下午一点
return: list of names and the times where their swipe badges within one hour. if there are multiple intervals that satisfy the condition, return any one of them.
name1: time1, time2, time3...
name2: time1, time2, time3, time4, time5...
example:
input: [['James', '1300'], ['Martha', '1600'], ['Martha', '1620'], ['Martha', '1530']] 
output: {
'Martha': ['1600', '1620', '1530']
'''
import collections
class Solution:
    def alertNames(self, keyName: list[str], keyTime: list[str]) -> list[str]:
        def is_within_1h(t1,t2):
            h1,m1=t1.split(":")
            h2,m2=t2.split(":")
            if int(h1)+1<int(h1):return False
            if h1==h2:return True
            return m1>=m2

        records=collections.defaultdict(list)
        for name,time in zip(keyName,keyTime):
            records[name].append(time)

        result=[]

        for person,record in records.items():
            record.sort()
            if any(is_within_1h(t1,t2) for t1,t2 in zip(record,record[2:])):
                result.append(person)
        return sorted(result)        

'''
我们需要实现一个功能，判断一个新的会议是否可以安排到当前的会议日程中。
输入：
meetings：一个二维列表，每个元素是 [start, end]，表示会议的开始时间和结束时间。
start 和 end：表示新的会议的开始和结束时间。
输出：

返回 True：如果新的会议在现有会议中没有冲突。
返回 False：如果新的会议和现有会议有时间上的冲突。
注意：

时间格式为整数，例如：
13:00 表示为 1300。
9:30 表示为 930。
冲突的定义是：
两个会议的时间有重叠。例如，现有会议 [930, 1200] 和新会议 [1100, 1230] 存在冲突。'''
def can_schedule_meeting(meetings,start,end):
    for meeting_start,meeting_end in meetings:
        if not (end<=meeting_start or start>=meeting_end):
            return False
    return True        
'''
我们需要实现一个函数，给定一组会议时间段，找到这些会议时间段之间的空闲时间段（包括从 0 开始到第一个会议的 start 的时间段）。

输入：
meetings：一个二维数组，每个元素 [start, end] 表示会议的开始时间和结束时间。
例如：[[1, 3], [5, 6], [2, 4]]
输出：
一个二维数组，每个元素 [start, end] 表示空闲时间段。'''
def spare_time(meetings):
    def merge_meetings(intervals):
        meetings.sort(key=lambda x:x[0])
        merged=[intervals[0]]
        for i in range(1,len(intervals)):
            prev_start,prev_end=merged[-1]
            curr_start,curr_end=intervals[i]

            if curr_start<=prev_end:
                merged[-1][-1]=max(prev_end,curr_end)
            else:
                merged.append([curr_start,curr_end])
        return merged
    if not meetings:
        return []
    merged_meetings=merge_meetings(meetings)
    result=[]
    start=0
    for meeting in merged_meetings:
        if start<meeting[0]:
            result.append([start,meeting[0]])
        start=meeting[1]
    return result        

'''
稀疏向量（Sparse Vector）是一种用于高效存储和操作大型向量的方法，其中大部分元素为零。
支持以下功能：
set(index, value)：设置指定索引的值。如果索引超出范围，则抛出 IndexOutOfBoundsException。
get(index)：获取指定索引的值。如果索引超出范围，则抛出 IndexOutOfBoundsException；如果索引未设置，返回 0.0。
toString()：返回向量的字符串表示，显示所有值，包括零。
索引超出范围时抛出错误：
当访问的索引超出稀疏向量的大小时，抛出 IndexOutOfBoundsException。
只存储非零值，以节省内存。
'''
class IndexOutOfBoundsException(Exception):
    """自定义索引越界异常"""
    pass

class Node:
    """链表节点类，存储稀疏向量中的非零值"""
    def __init__(self, val=0.0, index=0, next=None):
        self.val = val
        self.index = index
        self.next = next

class SparseVector:
    def __init__(self, size):
        """初始化稀疏向量，指定大小"""
        self.size = size
        self.head = None  # 链表头部

    def set(self, index, val):
        """设置指定索引的值"""
        if index >= self.size or index < 0:
            raise IndexOutOfBoundsException(f"Index {index} is out of range for size {self.size}.")
        
        # 处理链表插入逻辑
        if not self.head:
            # 如果链表为空，直接创建第一个节点
            self.head = Node(val, index)
            return
        
        # 遍历链表寻找插入点
        prev = None
        curr = self.head
        while curr and curr.index < index:
            prev = curr
            curr = curr.next
        
        if curr and curr.index == index:
            # 索引已存在，更新值
            curr.val = val
        else:
            # 插入新节点
            new_node = Node(val, index, curr)
            if prev:
                prev.next = new_node
            else:
                self.head = new_node  # 插入到链表头部

    def get(self, index):
        """获取指定索引的值"""
        if index >= self.size or index < 0:
            raise IndexOutOfBoundsException(f"Index {index} is out of range for size {self.size}.")
        
        curr = self.head
        while curr:
            if curr.index == index:
                return curr.val
            curr = curr.next
        return 0.0  # 如果索引未设置，返回 0.0

    def __str__(self):
        """返回稀疏向量的字符串表示"""
        result = []
        curr = self.head
        for i in range(self.size):
            if curr and curr.index == i:
                result.append(curr.val)
                curr = curr.next
            else:
                result.append(0.0)
        return "[" + ", ".join(map(str, result)) + "]"

'''
我们有一个二维网格，1 表示障碍物或不可通行的格子，0 表示可以通行的格子。任务是：

给定当前所在格子的坐标 (i, j)。
找到从当前位置出发，可以移动到的所有 上下左右 格子（0 表示可移动）。
输出这些可移动格子的坐标。'''
def find_legal_moves(grid,i,j):
    directions=[(-1,0),(1,0),(0,-1),(0,1)]
    legal_moves=[]
    rows,cols=len(grid),len(grid[0])

    for di,dj in directions:
        ni,nj=i+di,j+dj
        if 0<=ni<rows and 0<=nj<cols and grid[ni][nj]==0:
            legal_moves.append((ni,nj))
    return legal_moves        

'''
给定一个二维矩阵 matrix：

-1 表示墙（不能通过）。
0 表示可行的路径。
目标是判断是否能从给定的起点 (i, j)（其值为 0）出发，访问到矩阵中所有的 0。'''
def can_reach_all_zeros(matrix,i,j):
    if not matrix or not matrix[0]:
        return False
    if matrix[i][j] != 0:
        return False    
    rows,cols=len(matrix),len(matrix[0])
    visited=[[False for _ in range(cols)] for _ in range(rows)]

    def flood_fills_dfs(x,y):
        if (x<0 or x>=rows or y<0 or y>=cols or matrix[x][y]==-1
        or visited[x][y]):
            return
        visited[x][y]=True
        flood_fills_dfs(x-1,y)
        flood_fills_dfs(x+1,y)
        flood_fills_dfs(x,y-1)
        flood_fills_dfs(x,y+1)
    flood_fills_dfs(i,j)

    for x in range(rows):
        for y in range(cols):
            if matrix[x][y]==0 and not visited[x][y]:
                return False
    return True                

'''
board3 中1代表钻石，给出起点和终点，问有没有一条不走回头路的路线，能从起点走到终点，并拿走所有的钻石，给出所有的最短路径。

board3 = [
    [  1,  0,  0, 0, 0 ],
    [  0, -1, -1, 0, 0 ],
    [  0, -1,  0, 1, 0 ],
    [ -1,  0,  0, 0, 0 ],
    [  0,  1, -1, 0, 0 ],
    [  0,  0,  0, 0, 0 ],
]'''
def find_all_treasures(board,start,end):
    if not board:
        return []
    num_treasures=sum(row.count(1) for row in board)

    rows,cols=len(board),len(board[0])
    paths=[]
    def dfs(x,y,path,remaining_treasures):
        if (x<0 or x>=rows or y<0 
        or y>= cols or board[x][y]==-1 or board[x][y]==2):
            return
        path.append((x,y))
        temp=board[x][y]

        if temp==1:
            remaining_treasures-=1

        if (x,y)==end and remaining_treasures==0:
            paths.append(list(path))
            path.pop()
            return
        board[x][y]=2

        dfs(x+1,y,path,remaining_treasures)
        dfs(x-1,y,path,remaining_treasures)
        dfs(x,y+1,path,remaining_treasures)
        dfs(x,y-1,path,remaining_treasures)

        board[x][y]=temp  
        path.pop()
    dfs(start[0],start[1],[],num_treasures)

    if not paths:
        return
    min_length=min(len(path) for path in paths) 
    return [path for path in paths if len(path)==min_length]

'''Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according
 to the following rules:
Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.'''
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        N=9
        rows=[set() for _ in range(N)]
        cols=[set() for _ in range(N)]
        boxes=[set() for _ in range(N)]

        for r in range(N):
            for c in range(N):
                val=board[r][c]
                if val=='.':
                    continue
                if val in rows[r]:
                    return False
                rows[r].add(val)

                if val in cols[c]:
                    return False
                cols[c].add(val)

                idx=(r//3)*3+c//3
                if val in boxes[idx]:
                    return False
                boxes[idx].add(val)
        return True                
'''
输入:
image: 一个 m x n 的二维网格，其中 image[i][j] 是像素值。
sr 和 sc: 起始像素的行和列索引。
color: 需要填充的目标颜色。
任务: 从像素 image[sr][sc] 开始，使用 目标颜色 color 进行"泛洪填充"（Flood Fill），
规则如下：
起始像素的颜色改变为目标颜色。
改变所有与起始像素通过横向或纵向连接的像素（并且这些像素的原始颜色和起始像素颜色相同）。
最终返回修改后的图像。'''
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        # 获取行数和列数
        R, C = len(image), len(image[0])
        # 获取起始点的颜色
        color = image[sr][sc]
        # 如果目标颜色与起始颜色相同，直接返回原图
        if color == newColor:
            return image
        
        # 定义深度优先搜索 (DFS) 函数
        def dfs(r, c):
            # 如果当前位置的颜色与起始颜色相同，进行填充
            if image[r][c] == color:
                # 修改当前位置颜色为新颜色
                image[r][c] = newColor
                # 递归检查上、下、左、右四个方向
                if r >= 1:  # 上方像素
                    dfs(r - 1, c)
                if r + 1 < R:  # 下方像素
                    dfs(r + 1, c)
                if c >= 1:  # 左方像素
                    dfs(r, c - 1)
                if c + 1 < C:  # 右方像素
                    dfs(r, c + 1)

        # 从起始像素开始执行深度优先搜索
        dfs(sr, sc)
        # 返回修改后的图像
        return image
'''
A certain bug's home is on the x-axis at position x. Help them get there from position 0.

The bug jumps according to the following rules:

It can jump exactly a positions forward (to the right).
It can jump exactly b positions backward (to the left).
It cannot jump backward twice in a row.
It cannot jump to any forbidden positions.
The bug may jump forward beyond its home, but it cannot jump to positions numbered with
 negative integers.
Given an array of integers forbidden, where forbidden[i] means that the bug cannot jump to 
the position forbidden[i], and integers a, b, and x, return the minimum number of jumps 
needed for the bug to reach its home. If there is no possible sequence of jumps that lands 
the bug on position x, return -1.'''  
class Solution:
    def minimumJumps(self, forbidden: List[int], a: int, b: int, x: int) -> int:
        limit = 2000 + a + b
        visited = set(forbidden)
        myque = collections.deque([(0, True)]) # (pos, isForward) 
        hops = 0
        while(myque):
            l = len(myque)
            while(l > 0):
                l -= 1
                pos, isForward = myque.popleft()
                if pos == x:
                    return hops
                if pos in visited: continue
                visited.add(pos)
                if isForward:
                    nxt_jump = pos - b
                    if nxt_jump >= 0:
                        myque.append((nxt_jump, False))
                nxt_jump = pos + a
                if nxt_jump <= limit:
                    myque.append((nxt_jump, True))
            hops += 1
        return -1  