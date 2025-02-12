factorial(int):
  push    rbp
  mov     rbp, rsp
  push    rbx
  sub     rsp, 24
  mov     DWORD PTR [rbp-20], edi
  cmp     DWORD PTR [rbp-20], 0
  jne     .L2
  mov     eax, 1
  jmp     .L3
.L2:
  mov     eax, DWORD PTR [rbp-20]
  movsx   rbx, eax
  mov     eax, DWORD PTR [rbp-20]
  sub     eax, 1
  mov     edi, eax
  call    factorial(int)
  imul    rax, rbx
.L3:
  mov     rbx, QWORD PTR [rbp-8]
  leave
  ret
