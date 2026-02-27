#include <ATen/cuda/CUDAContext.h>

#include <memory>

#include "kernels/exception.cuh"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const at::cuda::CUDAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);  // 在 stream 中记录 event
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const { 
        // 即使 event 没有记录在 currentCUDAStream 中，也可以通过wait来等待event完成，即跨流等待：一个流可以等待另一个流中记录的事件。
        at::cuda::getCurrentCUDAStream().unwrap().wait(*event); 
    }
};

torch::Event create_event(const at::cuda::CUDAStream& s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

/*
执行这个函数就是相当于: 
    stream s_0 如果要执行 “当前时刻之后分配在 stream s_0 中的任务”，就需要等待 stream s_1 中的 “在当前时刻之前已经分配到 stream s_1 中的任务” 都执行完成。
这里的“当前时刻”包含了create_event和wait这两步，缺一不可。

注意：这是主机端函数，直接操作指定的 stream 对象，不涉及当前流，即使当前流不是s_0也不是s_1，也可以调用这个函数。
    */
void stream_wait(const at::cuda::CUDAStream& s_0, const at::cuda::CUDAStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

void stream_wait(const at::cuda::CUDAStream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);
}

}  // namespace deep_ep
