// A line-for-line loop-order port of pro_paired_environment.exact_potts_cut.
// Double residuals, strict cap>0, iterative DFS, no fast-math/quantization.
#include <vector>
#include <deque>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <cmath>
struct Edge { int target; int reverse; double capacity; };
extern "C" int e_float_cut(std::int64_t count,std::int64_t ec,const double *source_cost,
 const double *sink_cost,const std::int64_t *pairs,const double *caps,
 unsigned char *labels,double *flow_out,std::int64_t *phases_out) {
 try {
  const int n=static_cast<int>(count),source=n,sink=n+1;
  std::vector<std::vector<Edge>> graph(n+2);
  std::vector<std::int64_t> degree(n+2,0);degree[source]=n;degree[sink]=n;
  for(int i=0;i<n;++i)degree[i]=2;
  for(std::int64_t j=0;j<ec;++j){degree[pairs[2*j]]+=2;degree[pairs[2*j+1]]+=2;}
  for(int i=0;i<n+2;++i)graph[i].reserve(degree[i]);
  auto add=[&](int l,int r,double cap){
   graph[l].push_back({r,static_cast<int>(graph[r].size()),cap});
   graph[r].push_back({l,static_cast<int>(graph[l].size())-1,0.});
  };
  for(int i=0;i<n;++i){add(source,i,source_cost[i]);add(i,sink,sink_cost[i]);}
  for(std::int64_t j=0;j<ec;++j){int l=pairs[2*j],r=pairs[2*j+1];add(l,r,caps[j]);add(r,l,caps[j]);}
  double flow=0.;std::int64_t phases=0;
  std::vector<int> level(n+2),pointer(n+2),vertices;
  std::vector<std::pair<int,int>> path;vertices.reserve(n+2);path.reserve(n+2);
  while(true){
   std::fill(level.begin(),level.end(),-1);level[source]=0;std::deque<int> queue;queue.push_back(source);
   while(!queue.empty()){
    int vertex=queue.front();queue.pop_front();
    for(const Edge &e:graph[vertex])if(e.capacity>0.&&level[e.target]<0){level[e.target]=level[vertex]+1;queue.push_back(e.target);}
   }
   if(level[sink]<0)break;++phases;std::fill(pointer.begin(),pointer.end(),0);vertices.clear();path.clear();vertices.push_back(source);
   while(!vertices.empty()){
    int vertex=vertices.back();
    if(vertex==sink){
     double amount=std::numeric_limits<double>::infinity();
     for(const auto &p:path)amount=std::min(amount,graph[p.first][p.second].capacity);
     for(const auto &p:path){Edge &e=graph[p.first][p.second];e.capacity=std::max(0.,e.capacity-amount);graph[e.target][e.reverse].capacity+=amount;}
     flow+=amount;vertices.clear();path.clear();vertices.push_back(source);continue;
    }
    while(pointer[vertex]<static_cast<int>(graph[vertex].size())){
     const Edge &e=graph[vertex][pointer[vertex]];
     if(e.capacity>0.&&level[e.target]==level[vertex]+1)break;
     ++pointer[vertex];
    }
    if(pointer[vertex]==static_cast<int>(graph[vertex].size())){
     level[vertex]=-1;vertices.pop_back();
     if(!path.empty()){int previous=path.back().first;path.pop_back();++pointer[previous];}
    }else{path.push_back({vertex,pointer[vertex]});vertices.push_back(graph[vertex][pointer[vertex]].target);}
   }
  }
  std::vector<unsigned char> reachable(n+2,0);reachable[source]=1;std::deque<int> queue;queue.push_back(source);
  while(!queue.empty()){
   int vertex=queue.front();queue.pop_front();
   for(const Edge &e:graph[vertex])if(e.capacity>0.&&!reachable[e.target]){reachable[e.target]=1;queue.push_back(e.target);}
  }
  for(int i=0;i<n;++i)labels[i]=reachable[i];*flow_out=flow;*phases_out=phases;return 0;
 }catch(...){return 1;}
}
