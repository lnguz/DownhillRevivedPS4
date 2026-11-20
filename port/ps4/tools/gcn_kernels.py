"""The PS4's GPU kernels (port/ps4/kernels/*.cl) as one C header for port/ps4/gnm.c.

Each OpenCL C file is compiled by LLVM's AMDGPU backend for the PS4's GPU (GCN 1.1, gfx701) as an
HSA code object (amdgcn-amd-amdhsa) and linked into a shared object; the header holds its loadable
image (the PT_LOAD segments at their addresses, the gaps zeroed) and, per kernel, the offset of its
kernel descriptor (KERNEL.kd), from which gnm.c takes the entry point and the COMPUTE_PGM_RSRC1/2
words. No Sony shader compiler is involved: the GPU runs LLVM's code, dispatched by PM4 packets.

Usage: gcn_kernels.py <output header> [clang] [ld.lld]. Prints the header's SHA-256 (the build
puts it into a define, so the object cache sees a change of kernel)."""
import hashlib
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

KERNELS = Path(__file__).resolve().parents[1] / 'kernels'


def elf_image(data):
    """The loadable image of an ELF64 shared object and its symbols {name: value}."""
    (phoff, shoff) = struct.unpack_from('<QQ', data, 0x20)
    (phentsize, phnum, shentsize, shnum) = struct.unpack_from('<HHHH', data, 0x36)
    segments = []
    for i in range(phnum):
        p_type, _flags, offset, vaddr, _paddr, filesz, memsz, _align = struct.unpack_from('<IIQQQQQQ', data, phoff + i * phentsize)
        if p_type == 1:
            segments.append((vaddr, data[offset:offset + filesz], memsz))
    end = max(v + m for v, _, m in segments)
    image = bytearray(end)
    for vaddr, content, _ in segments:
        image[vaddr:vaddr + len(content)] = content
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize) for i in range(shnum)]
    symbols = {}
    for s in sections:
        if s[1] != 11:  # SHT_DYNSYM
            continue
        strtab = sections[s[6]]
        for at in range(s[4], s[4] + s[5], 24):
            name, _info, _other, _shndx, value, _size = struct.unpack_from('<IBBHQQ', data, at)
            text = data[strtab[4] + name:data.index(b'\0', strtab[4] + name)].decode()
            if text:
                symbols[text] = value
    return bytes(image), symbols


def graphics(tmp, llc):
    """The graphics shaders: (code, [(name, offset, {register: value})]) from kernels/graphics/*.ll,
    each file's .text placed at a 256-byte boundary after the last; the registers are LLVM's
    .AMDGPU.config pairs (SPI_SHADER_PGM_RSRC1/2_*, SPI_PS_INPUT_ENA/ADDR, ...), one block per
    function in the order of the functions."""
    code = bytearray()
    shaders = []
    for source in sorted((KERNELS / 'graphics').glob('*.ll')):
        obj = Path(tmp) / (source.stem + '.gfx.o')
        module = str(source)
        body = source.with_suffix('.cl')
        if body.exists():
            # the stage's body in OpenCL C, linked into the entry points and inlined there
            tool = lambda name: str(Path(llc).with_name(name)) if '/' in llc else name
            bc = Path(tmp) / (source.stem + '.body.bc')
            subprocess.run([tool('clang'), '-target', 'amdgcn-mesa-mesa3d', '-mcpu=gfx701', '-nogpulib', '-x', 'cl',
                            # no -cl-fp32-correctly-rounded-divide-sqrt here: its attributes differ from the
                            # entry points' and the body would not be inlined into them
                            # nor flushed denormals (OpenCL's default): the entry points keep IEEE ones; and
                            # the IEEE mode bit off, as graphics stages run (AMDGPU will not inline across it)
                            '-cl-std=CL1.2', '-ffp-contract=off', '-fdenormal-fp-math=ieee', '-mno-amdgpu-ieee',
                            '-fno-honor-nans', '-O2',
                            '-I', str(KERNELS.parent), '-emit-llvm', '-c', str(body), '-o', str(bc)], check=True)
            linked = Path(tmp) / (source.stem + '.linked.bc')
            subprocess.run([tool('llvm-link'), str(source), str(bc), '-o', str(linked)], check=True)
            optimized = Path(tmp) / (source.stem + '.opt.bc')
            # only the entry points stay: the body is inlined into them and then dropped (a function
            # left in the object would take one of the .AMDGPU.config blocks' places)
            entries = re.findall(r'define\s+amdgpu_\w+\s+\S+\s+@(\w+)', source.read_text())
            subprocess.run([tool('opt'), '-mtriple=amdgcn-mesa-mesa3d', '-mcpu=gfx701',
                            '-internalize-public-api-list=' + ','.join(entries), '-passes=internalize,default<O3>,globaldce',
                            str(linked), '-o', str(optimized)], check=True)
            module = str(optimized)
        subprocess.run([llc, '-mtriple=amdgcn-mesa-mesa3d', '-mcpu=gfx701', '-mattr=-flat-for-global', '-O3',
                        '-filetype=obj', module, '-o', str(obj)], check=True)
        data = obj.read_bytes()
        (shoff,) = struct.unpack_from('<Q', data, 0x28)
        (shentsize, shnum, shstrndx) = struct.unpack_from('<HHH', data, 0x3A)
        sections = [struct.unpack_from('<IIQQQQIIQQ', data, shoff + i * shentsize) for i in range(shnum)]
        names = sections[shstrndx]

        def name(s):
            at = names[4] + s[0]
            return data[at:data.index(b'\0', at)].decode()
        by = {name(s): s for s in sections}
        text = data[by['.text'][4]:by['.text'][4] + by['.text'][5]]
        config = data[by['.AMDGPU.config'][4]:by['.AMDGPU.config'][4] + by['.AMDGPU.config'][5]]
        symtab = by['.symtab']
        strtab = sections[symtab[6]]
        functions = []
        for at in range(symtab[4], symtab[4] + symtab[5], 24):
            nm, info, _other, _shndx, value, _size = struct.unpack_from('<IBBHQQ', data, at)
            if info & 15 == 2:  # STT_FUNC
                functions.append((value, data[strtab[4] + nm:data.index(b'\0', strtab[4] + nm)].decode()))
        functions.sort()
        blocks, block = [], None
        for at in range(0, len(config), 8):
            reg, value = struct.unpack_from('<II', config, at)
            if reg in (0xB028, 0xB128, 0xB228, 0xB328, 0xB428, 0xB528, 0xB848):  # a stage's RSRC1: a new function
                block = {}
                blocks.append(block)
            if block is not None:
                block[reg] = value
        base = (len(code) + 255) & ~255
        code.extend(b'\0' * (base - len(code)))
        code.extend(text)
        for (offset, fn), regs in zip(functions, blocks):
            shaders.append((fn, base + offset, regs))
    return bytes(code), shaders


def build(output, clang='clang', lld='ld.lld'):
    sources = sorted(KERNELS.glob('*.cl'))
    with tempfile.TemporaryDirectory() as tmp:
        objects = []
        for source in sources:
            obj = Path(tmp) / (source.stem + '.o')
            subprocess.run([clang, '-target', 'amdgcn-amd-amdhsa', '-mcpu=gfx701', '-nogpulib', '-cl-std=CL1.2',
                            '-cl-fp32-correctly-rounded-divide-sqrt', '-I', str(KERNELS.parent),
                            '-O3', '-mcode-object-version=5', '-ffp-contract=off',
                            # global memory through MUBUF (buffer_load addr64), as console shaders do: the
                            # PS4 sets up no FLAT apertures, and a FLAT access hung the probe (01.12)
                            '-Xclang', '-target-feature', '-Xclang', '-flat-for-global', '-c', str(source), '-o', str(obj)],
                           check=True)
            objects.append(str(obj))
        llc = str(Path(clang).with_name('llc')) if '/' in clang else 'llc'
        gfx_code, gfx_shaders = graphics(tmp, llc)
        so = Path(tmp) / 'kernels.so'
        subprocess.run([lld, '-shared', *objects, '-o', str(so)], check=True)
        image, symbols = elf_image(so.read_bytes())
    kernels = sorted(name[:-3] for name in symbols if name.endswith('.kd'))
    report = []
    for name in kernels:
        at = symbols[name + '.kd']
        private, = struct.unpack_from('<I', image, at + 4)
        rsrc1, = struct.unpack_from('<I', image, at + 48)
        if private:
            raise SystemExit('gcn_kernels: %s needs %d bytes of scratch memory, which gnm.c does not set up' % (name, private))
        report.append('%s: %d VGPRs, %d SGPRs' % (name, ((rsrc1 & 63) + 1) * 4, (((rsrc1 >> 6) & 15) + 1) * 8))
    print('; '.join(report), file=sys.stderr)
    lines = ['/* Generated by port/ps4/tools/gcn_kernels.py from port/ps4/kernels: do not edit. */',
             '#include <stdint.h>',
             'static const uint8_t gcn_image[%d] __attribute__((aligned(256))) = {' % len(image)]
    for at in range(0, len(image), 24):
        lines.append('    ' + ', '.join('%d' % b for b in image[at:at + 24]) + ',')
    lines.append('};')
    lines.append('static const struct { const char *name; uint32_t descriptor; } gcn_kernels[] = {')
    for name in kernels:
        lines.append('    {"%s", 0x%x},' % (name, symbols[name + '.kd']))
    lines.append('};')
    lines.append('static const uint8_t gcn_graphics[%d] __attribute__((aligned(256))) = {' % max(len(gfx_code), 1))
    for at in range(0, max(len(gfx_code), 1), 24):
        lines.append('    ' + ', '.join('%d' % b for b in (gfx_code or b'\0')[at:at + 24]) + ',')
    lines.append('};')
    lines.append('/* a graphics shader: its code offset in gcn_graphics, RSRC1/RSRC2, SPI_PS_INPUT_ENA/ADDR */')
    lines.append('static const struct { const char *name; uint32_t offset, rsrc1, rsrc2, ps_input_ena, ps_input_addr; } gcn_shaders[] = {')
    for fn, offset, regs in gfx_shaders:
        rsrc1 = regs.get(0xB028, regs.get(0xB128, 0))
        rsrc2 = regs.get(0xB02C, regs.get(0xB12C, 0))
        lines.append('    {"%s", 0x%x, 0x%x, 0x%x, 0x%x, 0x%x},' % (fn, offset, rsrc1, rsrc2, regs.get(0x286CC, 0), regs.get(0x286D0, 0)))
    lines.append('};')
    text = '\n'.join(lines) + '\n'
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


if __name__ == '__main__':
    print(build(sys.argv[1], *sys.argv[2:4]))
