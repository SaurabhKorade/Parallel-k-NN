/*
 *  Author :
 *  Saurabh Korade (skorade1@binghamton.edu)
 *  State University of New York, Binghamton
 */

#include <fstream>
#include <thread>
#include <vector>
#include <algorithm>
#include <string.h>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

using namespace std;

class KNN{
	private:
		//Static test data for debugging
		//vector<vector<float>> points{{2,3},{3,5},{1,4},{6,9},{9,6},{7,2},{4,7},{7,8},{5,4},{8,1}};
		//vector<vector<float>> queries{{7,4},{2,5}};
		
		int npoints,maxThreads,thread_count = 0;
		unsigned long long training_file_id,query_file_id,query_dim,nqueries,maxDim,k;
		time_t before_exec, after_exec;
		double exec_time;	
		
		vector<vector<float>> points;
		vector<vector<float>> queries;

		//Structure for storing the node of KD tree
		struct KDNode{
			vector<float> point;
			KDNode *left_child, *right_child;
		};
		//Structure to store query information
		struct QueryNode{
			vector<vector<float>> neighbors;
			vector<float> point;
			vector<float> euclidean_distance;
		};

		KDNode *root;
		QueryNode *qnode;

	public:
		const char *ptr;
	
		//Template to convert data from files
		template <typename T> KNN &operator>>(T &o){
			o = *(T *)ptr;
			ptr += sizeof(T);
			return *this;
		}		

		explicit KNN(const char *p) : ptr{p}{}
		
	public:
		/*KNN::KDNode(vector<float> p, KDNode *lc, KDNode *rc):point(move(p)), left_child(lc), right_child(rc){}*/
		
		KNN(unsigned long core_count,const char *training_file,const char *query_file,const char *output_file){
			cout << "Generating tree.Please wait.\n";
			before_exec = time(0);

			//Set threads count to double of the core count
			maxThreads = core_count * 2;
			read_file(training_file);
			
			//npoints=10;maxDim = 2;k=2;nqueries=2; For test data
			
			root = new KDNode;
			generate_tree(points,npoints,0,root);

			after_exec = time(0);
			exec_time = difftime(after_exec,before_exec);
			cout << "Execution time for generating tree: " << exec_time << "secs.\n";
			
			before_exec = time(0);
			read_file(query_file);
			if(maxDim == query_dim){
				qnode = new QueryNode[nqueries];
				for(int i =0; i < nqueries; i++){
					qnode[i].neighbors = vector<vector<float>>(k);
					qnode[i].euclidean_distance = vector<float>(k);
				}
				for(int i = 0; i < nqueries; i++){
					qnode[i].point = queries[i];
					find_nearest_neighbors(qnode[i],root,0,0);
				}
				
				write_result(output_file);
				
				after_exec = time(0);
				exec_time = difftime(after_exec,before_exec);
				cout << "Execution time for generating nearest neighbors: " << exec_time << "secs.\n";
			}else{
				cout << "Dimension of queries don't match the dimension of points. Enter valid query file.\n";
				print_query_point_data();
			}
		}
		
		/*
		 * Print the information of points and queries for debugging
		 */
		void print_query_point_data(){
			cout << "No of points:" << npoints << endl;
			cout << "Dimension of points:" << maxDim << endl;
			cout << "No of queries:" << nqueries << endl;
			cout << "Dimension of queries:" << query_dim << endl;
			cout << "k:" << k << endl;
		}
		
		/*
		 * Write result to output file
		 * @param output_file_path
		 */
		void write_result(const char *result_file_path){
			
			ifstream file(result_file_path);
			if (file) {
				cout << "Result file already exists. Provide another name.\n";
				exit(1);
			}
			unsigned char rand_buf[8];
			int fd = open("/dev/urandom", O_RDONLY);
			read(fd, rand_buf, 8);
			close(fd);
			
			unsigned long random_id = *reinterpret_cast<unsigned int *>(&rand_buf);
			
			ofstream result_file(result_file_path, ios::binary);

			result_file.write("RESULT\0\0", 8);
			result_file.write(reinterpret_cast<const char *> (&training_file_id),sizeof(this->training_file_id));
			result_file.write(reinterpret_cast<const char *> (&query_file_id),sizeof(this->query_file_id));
			result_file.write(reinterpret_cast<const char *> (&random_id),sizeof(random_id));
			result_file.write(reinterpret_cast<const char *> (&nqueries),sizeof(this->nqueries));
			result_file.write(reinterpret_cast<const char *> (&maxDim),sizeof(this->maxDim));
			result_file.write(reinterpret_cast<const char *> (&k),sizeof(this->k));
			
			for(int i = 0; i < nqueries; i++){
				for(int j = 0; j < k; j++){
					for(int k = 0; k < maxDim; k++){
						result_file.write(reinterpret_cast<const char *> (&qnode[i].neighbors[j][k]),sizeof(float));
					}
				}
			}
		}

		/*
		 * Prints points out a vector of vector
		 * @param 2d vector to be read
		 */		
		void print_data(vector<vector<float>> points){
			for (int i = 0; i < points.size(); i++){
				for (int j = 0; j < points[i].size(); j++){
					cout << points[i][j] << "\t";
				}
				cout << endl;
			}
		}
		
		/*
		 * Read quey file and training file data
		 * @param file_path
		 */
		void read_file(const char *file_name){
			int fd = open(file_name, O_RDONLY);
			if (fd < 0) {
				int en = errno;
				cerr << "Couldn't open " << file_name << ": " << strerror(en) << "." << endl;
				exit(2);
			}

			if (fd == -1){
				perror("Error opening file for writing");
				exit(EXIT_FAILURE);
			}

			struct stat fileInfo = {0};
			int rv = fstat(fd, &fileInfo);
			if (rv == -1){
				perror("Error getting the file size");
				exit(EXIT_FAILURE);
			}

			void *map = mmap(0, fileInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
			if (map == MAP_FAILED){
				close(fd);
				perror("Error mmapping the file");
				exit(EXIT_FAILURE);
			}
			rv = madvise(map, fileInfo.st_size, MADV_SEQUENTIAL | MADV_WILLNEED);
			
			char *file_mem = (char *)map;
			
			rv = close(fd);

			//Read type of file
			auto n = static_cast<unsigned long>(strnlen(file_mem, 8));
			string file_type(file_mem, n);
			
			// Start reading data and skip the file type
			KNN reader{file_mem + 8};
			
			if (file_type == "QUERY") {
				unsigned long id, n_queries, n_dims, n_neighbors;
				reader >> id >> n_queries >> n_dims >> n_neighbors;

				this->nqueries = n_queries;
				this->k = n_neighbors;
				this->query_file_id = id;
				this->query_dim = n_dims;

				for (unsigned long i = 0; i < n_queries; i++) {
					vector<float> dimensions;
					
					//(this->qnode[i]) = new QueryNode;
					for (unsigned long j = 0; j < n_dims; j++) {
						float f;
						reader >> f;
						dimensions.push_back(f);
					}
		
					this->queries.push_back(dimensions);
				}
			}else if (file_type == "TRAINING") {
				unsigned long long id, n_points, n_dims;
			
				reader >> id >> n_points >> n_dims;
				
				this->maxDim = n_dims;
				this->npoints = n_points;
				this->training_file_id = id;
				for (unsigned long i = 0; i < n_points; i++) {
					vector<float> dimensions;
			
					for (unsigned long j = 0; j < n_dims; j++) {
						float f;
						reader >> f;
						dimensions.push_back(f);
					}
					this->points.push_back(dimensions);
				}
			}
			//Read reuslt file to recheck answers
			else if (file_type == "RESULT") {
				unsigned long training_id, query_id, result_id, n_queries, n_dims, n_neighbors;
				float num;
				reader >> training_id >> query_id >> result_id >> n_queries >> n_dims >> n_neighbors;
		
				std::cout << "Training file ID: " << training_id << std::dec << std::endl;
				std::cout << "Query file ID: " << query_id << std::dec << std::endl;
				std::cout << "Result file ID: " << result_id << std::dec << std::endl;
				std::cout << "Number of queries: " << n_queries << std::endl;
				std::cout << "Number of dimensions: " << n_dims << std::endl;
				std::cout << "Number of neighbors returned for each query: " << n_neighbors << std::endl;
				int i=0;
				for(i = 0; i < nqueries*k*maxDim; i++){
					reader >> num;
					cout << num << endl;
				}
				cout << "itr" << i;

			}		
		}
		
		/*
		 * Generates a kd tree recursively spawning maxThreads
		 * @param points
		 * @param number_points
		 * @param dimension to work on
		 * @param node of the tree
		 */
		 void generate_tree(vector<vector<float>> points,int len,int dim, KDNode *n){
			
			int left_tree_len,right_tree_len,median;
			n->left_child = new KDNode;
			n->right_child = new KDNode;

			//Calculating the length of left and right subtree and position of median
			if(len % 2 == 0){
				left_tree_len = len / 2 - 1;
				median = len / 2 - 1;
			}else{
				left_tree_len = len / 2;
				median = len / 2;
			}
			right_tree_len = len /2;
			
			//Sort the data points according to dim dimension
			struct dimSort{
				dimSort(int dimx) { this->dimx = dimx; }
				bool operator ()( const vector<float>& points1, const vector<float>& points2 ) { 
					return points1[dimx] < points2[dimx]; 
				}
				int dimx;
			};
			sort(points.begin(), points.end(),dimSort(dim));
			
			//dimension variable to decide which dimension of the data to work on
			dim = (dim + 1) % maxDim;
			
			n->point = points[median];
			
			if(len>2){
				vector<vector<float>>::const_iterator l_first = points.begin();
				vector<vector<float>>::const_iterator l_last = points.begin() + median;
				vector<vector<float>> lt_median(l_first, l_last);
				
				vector<vector<float>>::const_iterator r_first = points.begin()+median+1;
				vector<vector<float>>::const_iterator r_last = points.end();
				vector<vector<float>> mt_median(r_first, r_last);
				
				//Spawn threas untill max count is reached 
				if(thread_count < maxThreads){
					thread th1(&KNN::generate_tree,this,lt_median,left_tree_len,dim,n->left_child);
					thread th2(&KNN::generate_tree,this,mt_median,right_tree_len,dim,n->right_child);
					thread_count += 2;
					th1.join();
					th2.join();
					thread_count -= 2;
				}else{
					generate_tree(lt_median,left_tree_len,dim,n->left_child);
					generate_tree(mt_median,right_tree_len,dim,n->right_child);
				}
			}else if(len == 1){
				n->left_child = NULL;
				n->right_child = NULL;
			}else if(len == 2){
				n->left_child = NULL;
				vector<vector<float>>::const_iterator r_first = points.begin()+median+1;
				vector<vector<float>>::const_iterator r_last = points.end();
				vector<vector<float>> mt_median(r_first, r_last);
				generate_tree(mt_median,right_tree_len,dim,n->right_child);
			}
		}
		
		/*
		 * Calcultes euclidean distance. Not performing square root because it is not required
		 * @param point1
		 * @param point2
		 * @return euclidean_distance
		 */
		float get_eucledian_distance(vector<float> &point1, vector<float> &point2){
			float ed = 0;
			
			if (point1.size() == point2.size()) {
				for(int i = 0; i<maxDim; i++){
					ed += (point1[i] - point2[i]) * (point1[i] - point2[i]);
				}
			}
			else{
				cout << "\tInvalid input to euclidean_distance()\t";
				exit(1);
			}
			
			return ed;
		}
		
		/*
		 * Finds the nearest neighbors for given query point
		 * @param query node 
		 * @param root of tree
		 * @param dimension to work on
		 * @param iterator
		 * Iterator blindly copies the first k nodes of the tree as the neighbors
		 */
		void find_nearest_neighbors(QueryNode &qnode, KDNode *kdnode, int dim, int itr){
			float temp,ed;
			int pos = 0;
			if(kdnode){
				ed = get_eucledian_distance(qnode.point, kdnode->point);
				
				//Store first k nodes as the nearest neighbors for comparison
				if(itr<k){
					qnode.neighbors.at(itr) = kdnode->point;
					qnode.euclidean_distance[itr] = ed;
					itr++;
				}
				
				//Find the node with highest euclidean distance and replace with new if new < highest
				else{
					temp = qnode.euclidean_distance[0];
					for(int i=0; i<=k; i++){
						if(qnode.euclidean_distance[i] > temp && i<k){
							temp = qnode.euclidean_distance[i];
							pos = i;
						}else if(i == k && pos < k && temp > ed){
							qnode.neighbors[pos] = kdnode->point;
							qnode.euclidean_distance[pos] = ed;
						}
					}
				}
				
				//Compare the dimension(dim) to decide the traversal path
				if(qnode.point[dim] < kdnode->point[dim]){kdnode = kdnode->left_child;}
				else{kdnode = kdnode->right_child;}
				dim = (dim + 1) % maxDim;
				find_nearest_neighbors(qnode,kdnode,dim,itr);
			}
		}

		/*
		 * Delete the tree after use
		 * @param root
		 */
		void delete_kdtree(KDNode *node){
			if(node){
				if(node->left_child){
					delete_kdtree(node->left_child);
				}
				if(node->right_child){
					delete_kdtree(node->right_child);
				}
				delete node;
			}
		}
		
		/*
		 * Destrcutor 
		~KNN(){
			delete_kdtree(root);
			delete qnode;
		}*/
};

int main(int argc, char **argv){
    if (argc != 5) {
        cerr << "Usage: "<<argv[0]<<" <n_cores> <training_file> <query_file> <result_file>" << endl;
        exit(1);
    }
	
    KNN init(strtoul(argv[1], nullptr, 10), argv[2], argv[3], argv[4]);
}
